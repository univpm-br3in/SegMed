"""Verify PyTorch's selected binary, GPU availability, and a 3D convolution."""
import argparse

import torch
import torchvision


WHEEL_TAGS = {
    "CPU": ("cpu", None),
    "CUDA126": ("cu126", "12.6"),
    "CUDA128": ("cu128", "12.8"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=WHEEL_TAGS)
    backend = parser.parse_args().backend
    wheel_tag, runtime_version = WHEEL_TAGS[backend]

    print(f"PyTorch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")

    if not torch.__version__.endswith("+" + wheel_tag):
        raise RuntimeError(f"Incorrect PyTorch wheel, expected +{wheel_tag}")
    if not torchvision.__version__.endswith("+" + wheel_tag):
        raise RuntimeError(f"Incorrect TorchVision wheel, expected +{wheel_tag}")

    if backend == "CPU":
        if torch.version.cuda is not None:
            raise RuntimeError("Expected the CPU-only PyTorch build")
        device = torch.device("cpu")
    else:
        if torch.version.cuda is None or not torch.version.cuda.startswith(runtime_version):
            raise RuntimeError(f"Expected a CUDA {runtime_version} PyTorch runtime")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access an NVIDIA GPU")
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {name}; compute capability: {capability[0]}.{capability[1]}")
        print(f"CUDA architectures in wheel: {torch.cuda.get_arch_list()}")

    a = torch.ones((2, 2), device=device)
    result = (a @ a).sum().item()
    if result != 8.0:
        raise RuntimeError("Matrix computation returned an unexpected result")

    # Exercise the same class of volumetric operator used by 3D segmentation.
    x = torch.ones((1, 1, 5, 5, 5), device=device)
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    result = torch.nn.functional.conv3d(x, kernel).sum().item()
    if result != 729.0:
        raise RuntimeError("3D convolution returned an unexpected result")
    if backend != "CPU":
        torch.cuda.synchronize()
    print(f"PyTorch {backend} verification passed")


if __name__ == "__main__":
    main()
