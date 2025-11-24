import os
import sys
import subprocess
from pathlib import Path


def nvcc_path():
    nvcc = os.environ.get("NVCC", "nvcc")
    return nvcc


def shared_ext():
    if os.name == "nt":
        return ".dll"
    elif sys.platform == "darwin":
        return ".dylib"
    return ".so"


def compile_cuda(src: Path, out: Path, arch: str = None):
    cmd = [nvcc_path(), "-std=c++14", "-Xcompiler", "-fPIC" if os.name != "nt" else "/MD", "-shared", "-o", str(out), str(src)]
    if arch:
        cmd.extend(["-arch", arch])
    include_dir = src.parent
    cmd.extend(["-I", str(include_dir)])
    print("Compiling", src.name)
    subprocess.check_call(cmd)


def main():
    root = Path(__file__).parent
    arch = os.environ.get("CUDA_ARCH", None)
    perceptron_src = root / "cepta_perceptron.cu"
    ssm_src = root / "cepta_ssm.cu"
    ext = shared_ext()
    compile_cuda(perceptron_src, root / f"cepta_perceptron{ext}", arch)
    compile_cuda(ssm_src, root / f"cepta_ssm{ext}", arch)
    print("Build finished.")


if __name__ == "__main__":
    main()
