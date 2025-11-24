import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


LIB_NAME = "cepta_ssm"


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


class LowRankCrossPathSSM:
    def __init__(self, P: int, Pr: int, hyperparams: Optional[dict] = None):
        self.lib = _load_library()
        self.lib.cepta_ssm_create.restype = ctypes.c_void_p
        self.lib.cepta_ssm_destroy.restype = None
        self.lib.cepta_ssm_forward.restype = None
        self.lib.cepta_ssm_backward.restype = None
        self.lib.cepta_ssm_update.restype = None
        self.lib.cepta_ssm_set_hyperparams.restype = None

        self.handle = ctypes.c_void_p(self.lib.cepta_ssm_create(P, Pr))
        if not self.handle:
            raise RuntimeError("Failed to create SSM state.")
        self.P = P
        self.Pr = Pr
        if hyperparams:
            self.set_hyperparams(**hyperparams)

    def set_hyperparams(
        self,
        lr_Vr: float = 1e-3,
        lr_Vb: float = 1e-3,
        lr_Vo: float = 1e-3,
        lr_Wl: float = 1e-3,
        lr_bl: float = 1e-3,
        a_min: float = 0.01,
        a_max: float = 0.99,
        eps_norm: float = 1e-6,
        use_norm: int = 1,
        scale_vb: int = 1,
    ):
        self.lib.cepta_ssm_set_hyperparams(
            self.handle,
            ctypes.c_float(lr_Vr),
            ctypes.c_float(lr_Vb),
            ctypes.c_float(lr_Vo),
            ctypes.c_float(lr_Wl),
            ctypes.c_float(lr_bl),
            ctypes.c_float(a_min),
            ctypes.c_float(a_max),
            ctypes.c_float(eps_norm),
            ctypes.c_int(use_norm),
            ctypes.c_int(scale_vb),
        )

    def forward(
        self, t_seq: np.ndarray, F_seq: np.ndarray, state0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        t_seq = np.ascontiguousarray(t_seq, dtype=np.float32)
        F_seq = np.ascontiguousarray(F_seq, dtype=np.int32)
        T = t_seq.shape[0]
        ttilde = np.empty_like(t_seq)
        state_out = np.empty((self.Pr,), dtype=np.float32)
        state0_ptr = None
        if state0 is not None:
            state0 = np.ascontiguousarray(state0, dtype=np.float32)
            state0_ptr = state0.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.cepta_ssm_forward(
            self.handle,
            t_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            F_seq.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(T),
            state0_ptr,
            ttilde.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            state_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return ttilde, state_out

    def backward(self, delta_ttilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        delta_ttilde = np.ascontiguousarray(delta_ttilde, dtype=np.float32)
        grad_state0 = np.empty((self.Pr,), dtype=np.float32)
        grad_t = np.empty((delta_ttilde.shape[0], self.P), dtype=np.float32)
        T = delta_ttilde.shape[0]
        self.lib.cepta_ssm_backward(
            self.handle,
            delta_ttilde.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(T),
            grad_state0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            grad_t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return grad_state0, grad_t

    def update(self):
        self.lib.cepta_ssm_update(self.handle)

    def __del__(self):
        try:
            if getattr(self, "handle", None):
                self.lib.cepta_ssm_destroy(self.handle)
        except Exception:
            pass


_lib = _load_library()
_lib.cepta_linear_forward.restype = None
_lib.cepta_linear_backward.restype = None
_lib.cepta_rmsnorm_forward.restype = None
_lib.cepta_rmsnorm_backward.restype = None


def linear_forward(X: np.ndarray, W: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    b_ptr = None
    if b is not None:
        b = np.ascontiguousarray(b, dtype=np.float32)
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B, in_dim = X.shape
    out_dim = W.shape[0]
    Y = np.empty((B, out_dim), dtype=np.float32)
    _lib.cepta_linear_forward(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_ptr,
        ctypes.c_int(B),
        ctypes.c_int(in_dim),
        ctypes.c_int(out_dim),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return Y


def linear_backward(X: np.ndarray, W: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(X, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    delta = np.ascontiguousarray(delta, dtype=np.float32)
    B, in_dim = X.shape
    out_dim = W.shape[0]
    grad_X = np.empty_like(X)
    grad_W = np.empty_like(W)
    grad_b = np.empty((out_dim,), dtype=np.float32)
    _lib.cepta_linear_backward(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        delta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
        ctypes.c_int(in_dim),
        ctypes.c_int(out_dim),
        grad_X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grad_W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grad_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return grad_X, grad_W, grad_b


def rmsnorm_forward(X: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(X, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    B, D = X.shape
    Y = np.empty_like(X)
    rms = np.empty((B,), dtype=np.float32)
    _lib.cepta_rmsnorm_forward(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
        ctypes.c_int(D),
        ctypes.c_float(eps),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rms.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return Y, rms


def rmsnorm_backward(
    X: np.ndarray, w: np.ndarray, delta: np.ndarray, rms: np.ndarray, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(X, dtype=np.float32)
    w = np.ascontiguousarray(w, dtype=np.float32)
    delta = np.ascontiguousarray(delta, dtype=np.float32)
    rms = np.ascontiguousarray(rms, dtype=np.float32)
    B, D = X.shape
    grad_X = np.empty_like(X)
    grad_w = np.empty_like(w)
    _lib.cepta_rmsnorm_backward(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        delta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rms.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(B),
        ctypes.c_int(D),
        ctypes.c_float(eps),
        grad_X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grad_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return grad_X, grad_w
