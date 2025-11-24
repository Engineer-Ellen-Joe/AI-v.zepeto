import ctypes
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


LIB_NAME = "cepta_perceptron"


def _shared_ext() -> str:
    if os.name == "nt":
        return ".dll"
    if os.name == "posix" and os.uname().sysname.lower() == "darwin":
        return ".dylib"
    return ".so"


def _load_library() -> ctypes.CDLL:
    root = Path(__file__).parent / "cuda"
    ext = _shared_ext()
    lib_path = root / f"{LIB_NAME}{ext}"
    if not lib_path.exists():
        build_script = root / "build.py"
        if not build_script.exists():
            raise RuntimeError("Missing build.py to compile CUDA kernels.")
        os.system(f'"{os.environ.get("PYTHON", "python")}" "{build_script}"')
        if not lib_path.exists():
            raise RuntimeError(f"Failed to build CUDA library at {lib_path}")
    return ctypes.CDLL(str(lib_path))


class CeptaPerceptron:
    def __init__(self, P: int, d: int, alpha: int, vocab: int, hyperparams: Optional[Dict] = None):
        self.lib = _load_library()
        self.lib.cepta_perceptron_create.restype = ctypes.c_void_p
        self.lib.cepta_perceptron_set_hyperparams.restype = None
        self.lib.cepta_perceptron_destroy.restype = None
        self.lib.cepta_forward_dense.restype = None
        self.lib.cepta_forward_index.restype = None
        self.lib.cepta_backward_dense.restype = None
        self.lib.cepta_backward_index.restype = None
        self.lib.cepta_update_params.restype = None

        self.handle = ctypes.c_void_p(self.lib.cepta_perceptron_create(P, d, alpha, vocab))
        if not self.handle:
            raise RuntimeError("Failed to create CeptaPerceptron state.")
        self.P = P
        self.d = d
        self.alpha = alpha
        self.vocab = vocab
        if hyperparams:
            self.set_hyperparams(**hyperparams)

    def set_hyperparams(
        self,
        eta_bp_w: float = 1e-3,
        eta_bp_f: float = 1e-3,
        eta_w: float = 1e-3,
        eta_f: float = 1e-3,
        eta_sp: float = 1e-3,
        w_min: float = -1.0,
        w_max: float = 1.0,
        f_min: float = -1.0,
        f_max: float = 1.0,
        sp_min: float = -5.0,
        sp_max: float = 5.0,
        z_ref: float = 1.0,
        y_ref: float = 1.0,
        eps_z: float = 1e-6,
        eps_y: float = 1e-6,
        beta_r: float = 0.1,
        beta_m: float = 0.1,
        r_target: float = 0.2,
        m_target: float = 0.2,
        lambda_r: float = 1.0,
        lambda_m: float = 1.0,
        mask_w_mode: int = 0,
        mask_f_mode: int = 0,
        use_ste: int = 0,
        ste_slope: float = 1.0,
    ):
        self.lib.cepta_perceptron_set_hyperparams(
            self.handle,
            ctypes.c_float(eta_bp_w),
            ctypes.c_float(eta_bp_f),
            ctypes.c_float(eta_w),
            ctypes.c_float(eta_f),
            ctypes.c_float(eta_sp),
            ctypes.c_float(w_min),
            ctypes.c_float(w_max),
            ctypes.c_float(f_min),
            ctypes.c_float(f_max),
            ctypes.c_float(sp_min),
            ctypes.c_float(sp_max),
            ctypes.c_float(z_ref),
            ctypes.c_float(y_ref),
            ctypes.c_float(eps_z),
            ctypes.c_float(eps_y),
            ctypes.c_float(beta_r),
            ctypes.c_float(beta_m),
            ctypes.c_float(r_target),
            ctypes.c_float(m_target),
            ctypes.c_float(lambda_r),
            ctypes.c_float(lambda_m),
            ctypes.c_int(mask_w_mode),
            ctypes.c_int(mask_f_mode),
            ctypes.c_int(use_ste),
            ctypes.c_float(ste_slope),
        )

    def forward_dense(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(X, dtype=np.float32)
        B = X.shape[0]
        Z = np.empty((B, self.P), dtype=np.float32)
        F = np.empty((B, self.P), dtype=np.int32)
        Y = np.empty((B, self.P, self.alpha), dtype=np.float32)
        t = np.empty((B, self.P), dtype=np.float32)
        self.lib.cepta_forward_dense(
            self.handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(B),
            Z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            F.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return Z, F, Y, t

    def forward_index(self, tok: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tok = np.ascontiguousarray(tok, dtype=np.int32)
        B = tok.shape[0]
        Z = np.empty((B, self.P), dtype=np.float32)
        F = np.empty((B, self.P), dtype=np.int32)
        Y = np.empty((B, self.P, self.alpha), dtype=np.float32)
        t = np.empty((B, self.P), dtype=np.float32)
        self.lib.cepta_forward_index(
            self.handle,
            tok.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(B),
            Z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            F.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return Z, F, Y, t

    def backward_dense(self, delta_Y: Optional[np.ndarray] = None, delta_t: Optional[np.ndarray] = None) -> np.ndarray:
        delta_y_ptr = None
        delta_t_ptr = None
        B = 0
        if delta_Y is not None:
            delta_Y = np.ascontiguousarray(delta_Y, dtype=np.float32)
            B = delta_Y.shape[0]
            delta_y_ptr = delta_Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if delta_t is not None:
            delta_t = np.ascontiguousarray(delta_t, dtype=np.float32)
            B = delta_t.shape[0]
            delta_t_ptr = delta_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        grad_X = np.empty((B, self.d), dtype=np.float32) if B > 0 else np.empty((0, self.d), dtype=np.float32)
        self.lib.cepta_backward_dense(
            self.handle,
            delta_y_ptr,
            delta_t_ptr,
            ctypes.c_int(B),
            grad_X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if B > 0 else None,
        )
        return grad_X

    def backward_index(self, delta_Y: Optional[np.ndarray] = None, delta_t: Optional[np.ndarray] = None):
        if delta_Y is not None:
            delta_Y = np.ascontiguousarray(delta_Y, dtype=np.float32)
            B = delta_Y.shape[0]
            delta_y_ptr = delta_Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            B = delta_t.shape[0] if delta_t is not None else 0
            delta_y_ptr = None
        delta_t_ptr = None
        if delta_t is not None:
            delta_t = np.ascontiguousarray(delta_t, dtype=np.float32)
            delta_t_ptr = delta_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.cepta_backward_index(
            self.handle,
            delta_y_ptr,
            delta_t_ptr,
            ctypes.c_int(B),
        )

    def update(self):
        self.lib.cepta_update_params(self.handle)

    def __del__(self):
        try:
            if getattr(self, "handle", None):
                self.lib.cepta_perceptron_destroy(self.handle)
        except Exception:
            pass
