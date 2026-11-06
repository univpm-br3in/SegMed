"""Verify SegMed runtime imports and the selected inference backend."""
import argparse
import importlib.util
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("CPU", "CUDA126", "CUDA128"))
    args = parser.parse_args()

    if sys.platform != "win32" or platform.architecture()[0] != "64bit":
        raise RuntimeError("This installer targets 64-bit Windows")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Expected Python 3.11, got {sys.version}")

    import torch
    import torchvision
    import tensorflow as tf
    import monai
    import PySide6

    for package in ("nnunet", "nnunetv2", "nibabel", "pyqtgraph"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(f"Missing dependency: {package}")

    if args.backend in ("CUDA126", "CUDA128") and not torch.cuda.is_available():
        raise RuntimeError("CUDA setup selected but PyTorch cannot use the GPU")
    if args.backend == "CPU" and torch.version.cuda is not None:
        raise RuntimeError("CUDA-enabled PyTorch installed in CPU environment")

    from modules.mainwindow import MainWindow

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"MONAI: {monai.__version__}")
    print(f"TensorFlow: {tf.__version__} (native Windows CPU)")
    print(f"PySide6: {PySide6.__version__}")
    print(f"MainWindow import: {MainWindow.__name__}")
    print("SegMed environment verification passed")


if __name__ == "__main__":
    main()
