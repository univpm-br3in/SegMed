# SegMed — Medical Image Segmentation Tool

[![Version](https://img.shields.io/badge/version-1.5.0-blue.svg)](https://github.com/univpm-br3in/SegMed/releases)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

![SegMed LOGO](resource/imgs/Segmed.png)
> **SegMed: An Open-Source Desktop Tool for Deploying Pretrained Deep Learning Models in 3D Medical Image Segmentation**

SegMed is an open-source desktop application for running pretrained deep learning segmentation models on 3D medical images, visualizing predictions, evaluating segmentations, and exporting anatomical surface meshes. It supports nnU-Net, MONAI, PyTorch, Keras, and TensorFlow.

**New in v1.5.0:** A Windows installer with automatic NVIDIA GPU configuration for supported PyTorch builds, TensorFlow SavedModel support, and improved image-to-model spatial shape handling.

**[Download SegMed for Windows](https://github.com/univpm-br3in/SegMed/releases/latest)** · [Install from source](#installation-from-source) · [Cite SegMed](#citation)


---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [GPU Acceleration](#gpu-acceleration)
- [Usage](#usage)
- [Supported Model Formats](#supported-model-formats)
- [Preprocessing Options](#preprocessing-options)
- [Visualization Features](#visualization-features)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 🧠 **Multi-Framework Model Support**
- **nnU-Net V1 & V2**: Full support for both versions including 2D and 3D models
- **PyTorch Models**: Support for `.pt`, `.pth`, `.pth.tar`, and TorchScript (`.ts`) models
- **Keras/TensorFlow Models**: Support for `.h5`, `.keras`, TensorFlow SavedModel directories (`saved_model.pb` and `variables/`), and supported architecture/weights formats
- **MONAI Bundles**: Complete support for MONAI bundle architectures

### 🖼️ **Advanced Image Processing**
- **Multiple Image Formats**: DICOM, NIfTI (`.nii`, `.nii.gz`), NRRD (`.nrrd`) and NumPy arrays (`.npy`)
- **DICOMDIR Support**: Load series from DICOMDIR files with series selection dialog
- **Window/Level Presets**: Pre-configured presets for brain, bone, lung, liver, and more
- **Custom Window/Level**: Manual adjustment for optimal visualization

### 🔧 **Preprocessing Pipeline**
- **Z-Normalization**: Image-based or manual (mean/std)
- **Min/Max Normalization**: Custom range normalization
- **Gaussian Blur**: Noise reduction with configurable sigma
- **Isotropic Resampling**: Uniform voxel spacing adjustment
- **Model Input Shape Adaptation**: Automatic zero-padding for smaller images and selectable cropping for larger images, with inverse spatial restoration of the predicted segmentation
- **Automatic Preprocessing**: Model-specific preprocessing for nnU-Net and MONAI models

### 📊 **Visualization & Analysis**

#### 2D Visualization
- **Three Orthogonal Views**: Axial, Sagittal, and Coronal
- **Interactive Sliders**: Navigate through all slices
- **Segmentation Overlay**: Color-coded class overlays with transparency
- **Comparison Mode**: 
  - Predicted segmentation as solid overlay
  - Ground truth as contour lines
  - Visual difference detection

#### 3D Visualization
- **Interactive 3D Mesh Rendering**: Real-time 3D visualization using PyQtGraph OpenGL
- **Multi-Class Display**: Individual mesh for each segmentation class
- **Color Customization**: 148 available colors with custom naming
- **Class Toggle**: Enable/disable individual classes
- **Comparison Mode**:
  - Ground truth: High transparency (20%)
  - Predicted: Medium opacity (70%)
  - Simultaneous visualization

#### Quantitative Metrics
- **DICE Coefficient**: Overlap similarity measurement
- **IoU (Intersection over Union)**: Jaccard Index calculation
- **Per-Class Metrics**: Individual metrics for each anatomical structure
- **Overall Metrics**: Aggregated performance across all classes
- **Real-time Display**: Metrics panel with automatic updates

### 💾 **Export Capabilities**
- **Segmentation Export**: Save predictions as NIfTI (.nii.gz) files
- **3D Mesh Export**: 
  - STL format (single or multi-file)
  - OBJ format (single or multi-file)
  - Per-class export with custom naming
- **Preserve Metadata**: Maintains spacing, origin, and orientation information



---

## 🚀 Installation

### Windows installer (recommended)

SegMed **v1.5.0** includes a Windows installer. You do not need to install Python, `uv`, PyTorch, or the other Python packages manually.

1. Open the [SegMed GitHub Releases page](https://github.com/univpm-br3in/SegMed/releases) and download the **SegMed Setup `.exe`** attached to the v1.5.0 release.
2. Run the installer. An internet connection is required to download Python and the necessary packages during the initial setup.
3. Optionally select **Automatically configure supported NVIDIA GPU acceleration**. The installer checks the GPU, driver, and compute capability, selects a compatible PyTorch CUDA build from its supported profiles, and tests GPU execution. If no supported configuration works, it installs CPU-only PyTorch.
4. When setup finishes, launch **SegMed** from the Start menu or its desktop shortcut (if selected).

The installer creates a private Python 3.11 environment for SegMed. It does not install the full CUDA Toolkit or modify your existing system Python installation. **An NVIDIA graphics driver must already be installed for GPU acceleration.** PyTorch-based inference (including nnU-Net and MONAI) can use the selected GPU when supported; TensorFlow inference uses the CPU in this native-Windows installer.

The GPU selector currently targets tested **PyTorch 2.8.0 CUDA 12.8 and 12.6** builds. Other NVIDIA configurations may use CPU instead. Successful GPU setup does not guarantee every model will fit in GPU memory.

If installation or dependency setup fails, see [Troubleshooting](#troubleshooting).

### Installation from source

Source installation remains available for researchers who want to manage or modify their dependencies. SegMed's Python version and dependencies are specified in `.python-version`, `pyproject.toml`, and `uv.lock`.

**Requirements:** Python 3.11 and the [`uv` package manager](https://docs.astral.sh/uv/). On Windows, install `uv` using its [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/univpm-br3in/SegMed.git
cd SegMed
uv sync --locked
uv run --no-sync python main.py
```

On the first `uv sync`, dependencies are downloaded and a local `.venv` environment is created. Run SegMed from the repository directory so application resources are found correctly.

**Manual GPU setup:** The source installation uses the packages resolved in `uv.lock`; it does **not** run the Windows installer's hardware-aware PyTorch selector. If you need a different PyTorch CUDA build, consult the [official PyTorch instructions](https://pytorch.org/get-started/locally/) and manage that environment explicitly. Subsequent `uv sync` or ordinary `uv run` operations can replace manually installed PyTorch wheels with the versions recorded in the lockfile. The Windows installer handles the supported CUDA profiles separately.

There is no `Start.bat` launcher. Run from source with `uv run --no-sync python main.py`, or use the Windows-installed shortcut.

---

## ⚡ GPU Acceleration

| Inference backend | Windows installer: CPU | Windows installer: NVIDIA GPU |
|---|---|---|
| PyTorch | Supported | Supported on compatible hardware |
| nnU-Net V1/V2 | Supported | Through CUDA-enabled PyTorch |
| MONAI | Supported | Through CUDA-enabled PyTorch |
| Keras / TensorFlow | Supported | **Not supported on native Windows** |

### PyTorch and NVIDIA GPUs

On Windows, the installer inspects NVIDIA hardware and the installed driver, then attempts a compatible precompiled PyTorch CUDA distribution. It verifies GPU operation before selecting that backend. If the GPU or driver does not match a supported configuration, or verification fails, installation falls back to CPU inference.

To verify PyTorch in a **source** installation:

```bash
uv run --no-sync python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For the **Windows installer**, use the Start menu's environment repair/reconfiguration entries if you need to retry GPU setup. Do not run `uv sync` inside the installed application directory, as it may replace the installer-selected CUDA wheels.

### TensorFlow on Windows: CPU only

**SegMed's native-Windows installer does not support GPU inference with TensorFlow.** This is a TensorFlow platform limitation, not a SegMed-specific error: **TensorFlow 2.10 was the last release with native-Windows NVIDIA CUDA support.** Official CUDA GPU support for TensorFlow 2.11 and later is provided through Linux or Windows Subsystem for Linux 2 (WSL2), rather than native Windows.

SegMed uses a newer TensorFlow version, so **Keras and TensorFlow SavedModel inference on native Windows runs on the CPU**, even if a compatible NVIDIA GPU is present and PyTorch can use it. WSL2/Linux GPU configurations are outside the scope of this Windows installer.

See [TensorFlow's official installation documentation](https://www.tensorflow.org/install/pip) for supported GPU environments.

To inspect TensorFlow devices in a source installation:

```bash
uv run --no-sync python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('Visible GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Note:** PyTorch and TensorFlow have independent GPU support. Enabling GPU acceleration for PyTorch does not enable TensorFlow GPU inference on native Windows.

---

## 🎮 Usage

### Basic Workflow

1. **Launch the Application**
   - Windows installer: open **SegMed** from the Start menu or desktop shortcut.
   - Source installation: run `uv run --no-sync python main.py` from the repository directory.


2. **Load a Neural Network Model**
   - Click "Load Network"
   - Select your model file (supports multiple formats)
   - Model configuration is auto-detected

3. **Load Medical Image**
   - Click "Load Image"
   - Choose DICOM, NIfTI, or image files
   - For DICOMDIR, select the desired series

4. **Optional: Load Ground Truth Segmentation**
   - Click "Load Segmentation"
   - Select ground truth NIfTI file
   - Enables comparison and metrics calculation

5. **Optional: Configure Preprocessing**
   - Click "Preprocessing Options!"
   - Select desired preprocessing steps
   - Configure parameters

6. **Run Segmentation**
   - Click "Segment!"
   - Select an available execution device (GPU support depends on the installed backend)
   - Wait for inference to complete

7. **Visualize Results**
   - Use radio buttons to switch between:
     - Loaded Segmentation (ground truth)
     - Predicted Segmentation
     - Compare Both (overlay + contours)
   - Navigate slices using sliders
   - Rotate 3D view with mouse

8. **Analyze Metrics**
   - View DICE and IoU scores in the metrics panel
   - Check per-class performance
   - Scroll to see all classes

9. **Customize Colors**
   - Click "Change Colors"
   - Select colors for each class
   - Name anatomical structures
   - Enable/disable classes

10. **Export Results**
    - Save segmentation: Click "Export segmentation"
    - Export 3D mesh: Click "Export Mesh"
    - Choose format (STL/OBJ) and mode (single/multi-file)

---

## 🤖 Supported Model Formats

### nnU-Net Models

#### nnU-Net V2
- **Location**: Trained model folder with `plans.json`
- **Configurations**: 2d, 3d_fullres, 3d_lowres, 3d_cascade
- **Features**: Auto-detection, automatic preprocessing

#### nnU-Net V1
- **Location**: Trained model folder with `plans.pkl`
- **Configurations**: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres
- **Features**: Fold ensemble, automatic preprocessing

### PyTorch Models
- **TorchScript**: `.ts` files
- **Checkpoints**: `.pt`, `.pth`, `.pth.tar` files
- **Requirements**: Model must have defined input shape

### Keras/TensorFlow Models
- **Keras Models**: `.h5` and `.keras` files
- **TensorFlow SavedModel**: A SavedModel directory containing `saved_model.pb` and its `variables/` directory; **the `.pb` file alone is not sufficient**
- **Architecture Files**: `.json` or `.yaml`, with the required weights
- **Model Requirements**: Custom layers/objects and expected preprocessing must be compatible with the installed TensorFlow version
- **Input Channels**: Standalone inference currently supports single-channel medical images; models requiring multiple distinct input channels are not supported automatically

**2D networks** can be applied slice by slice to a 3D image. SegMed does not currently provide a dedicated time-series/video segmentation workflow.

### MONAI Bundles
- **Structure**: Folder with `configs/` and `models/` directories
- **Config Files**: `inference.json` or `inference.yaml`
- **Features**: Automatic transform pipeline, sliding window inference

---

## 🔬 Preprocessing Options

### Z-Normalization (Image-Based)
Normalizes image using its own statistics:
```
normalized = (image - mean) / std
```

### Z-Normalization (Manual)
Normalizes using provided mean and standard deviation:
```
Input: mean, std
normalized = (image - mean) / std
```

### Min/Max Normalization
Scales image to specified range:
```
Input: min_val, max_val
normalized = (image - current_min) / (current_max - current_min) * (max_val - min_val) + min_val
```

### Gaussian Blur
Applies Gaussian smoothing:
```
Input: sigma (single value or per-axis)
```

### Isotropic Resampling
Resamples to uniform voxel spacing:
```
Input: target_spacing (or current + target)
```

**Note:** Isotropic resampling, when enabled, is performed before fitting images to a standalone model's input shape. Smaller spatial dimensions are zero-padded; larger dimensions require a crop position choice. The predicted segmentation is mapped back using the inverse crop/pad offsets, with nearest-neighbor interpolation when reversing resampling. Areas excluded by cropping cannot be recovered and remain background. Always use the preprocessing expected by the trained model.

---

## 📈 Visualization Features

### 2D Views
- **Axial**: Top-down view (slice along Z-axis)
- **Sagittal**: Side view (slice along X-axis)
- **Coronal**: Front view (slice along Y-axis)
- **Synchronized Navigation**: All views update together
- **Aspect Ratio Preservation**: True anatomical proportions

### 3D Rendering
- **Marching Cubes Algorithm**: High-quality surface extraction
- **Physical Coordinates**: Respects spacing, origin, and orientation
- **Interactive Controls**:
  - Rotate: Left mouse drag
  - Pan: Right mouse drag
  - Zoom: Mouse wheel

### Comparison Mode
When both ground truth and predicted segmentations are loaded:

**2D Views:**
- Predicted: Semi-transparent colored overlay
- Ground Truth: Bright contour outlines
- Easy identification of differences

**3D Views:**
- Ground Truth: Very transparent (ghosted)
- Predicted: Semi-solid (prominent)
- Visual overlap assessment

**Metrics Panel:**
- Automatic DICE and IoU calculation
- Overall and per-class scores
- Scrollable for many classes


---

## 🗂️ Project Structure

```
SegMed/
├── main.py                          # Application entry point
├── modules/
│   ├── mainwindow.py                # Main GUI window
│   ├── images.py                    # Image handling class
│   ├── mesh.py                      # 3D mesh generation
│   ├── networks.py                  # Neural network interface
│   ├── nnunet_handler.py            # nnU-Net V2 handler
│   ├── nnunet_v1_handler.py         # nnU-Net V1 handler
│   ├── preprocessingexecutor.py    # Preprocessing pipeline
│   ├── preprocessingwindow.py       # Preprocessing GUI
│   ├── colorselectionwindow.py      # Color picker dialog
│   ├── dicomSeriesSelectionDialog.py # DICOM series selector
│   └── networkdimensioninputwindow.py # Manual dimension input
├── resource/
│   └── imgs/
│       ├── icon.ico                 # Application icon
│       ├── Segmed.png              # Logo
│       └── univpm.png              # University logo
├── installer/                       # Windows installer source and setup scripts
│   ├── segmed.iss                   # Inno Setup configuration
│   ├── prepare_windows.ps1          # Download uv for installer compilation
│   ├── install_dependencies.ps1     # Dependency installation and GPU setup
│   └── select_torch_backend.py      # NVIDIA hardware detection
├── launcher.py                      # Installed Windows application launcher
├── pyproject.toml                   # Project dependencies
├── uv.lock                          # Locked dependency versions
├── .python-version                  # Python version
├── README.md                        # This file
└── LICENSE                          # License file
```



---

## 📚 Citation

If you use SegMed in your research, please cite the following paper:

```bash
@article{mortada2026segmed,
  author         = {Mortada, Mhd Jafar and Sbrollini, Agnese and Proniewska-van Dam, Klaudia and Van Dam, Peter M. and Burattini, Laura},
  title          = {{SegMed}: An Open-Source Desktop Tool for Deploying Pretrained Deep Learning Models in 3D Medical Image Segmentation},
  journal        = {Applied Sciences},
  volume         = {16},
  number         = {7},
  pages          = {3490},
  year           = {2026},
  doi            = {10.3390/app16073490},
  url            = {https://www.mdpi.com/2076-3417/16/7/3490},
  issn           = {2076-3417}
}
```
---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/univpm-br3in/SegMed.git
cd SegMed
uv sync --locked
uv run --no-sync python main.py
```

The installer build configuration is stored in `installer/`. See `installer/README_INSTALLER.md` for Windows packaging instructions. Avoid committing generated installer binaries, virtual environments, or build directories to the source repository.

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to new functions
- Comment complex logic
- Test with multiple model formats

---

## 📝 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This software is provided for **research and educational purposes only**.  
It is **not intended for clinical use**, **medical diagnosis**, **treatment planning**, or **decision-making**.  
The authors assume **no responsibility or liability** for any use of this software in clinical practice.

---

## 🙏 Acknowledgments

- nnU-Net team for the excellent segmentation framework
- MONAI team for medical imaging tools
- The open-source medical imaging community

---

## 🔄 Version History

### Version 1.5.0 (Current)
- Added a Windows installer with a private Python environment and automated dependency installation using `uv`.
- Added automatic NVIDIA GPU detection and selection of supported PyTorch CUDA 12.8/12.6 builds, with a CPU fallback.
- Added support for TensorFlow SavedModel directories (`saved_model.pb` and `variables/`).
- Improved spatial input-size handling through symmetric zero-padding, configurable cropping, and inverse restoration of segmentations.
- Improved isotropic-resampling order and segmentation restoration, including nearest-neighbor interpolation for masks.
- Removed the legacy `Start.bat` launcher; Windows users can use the installer and source users can launch through `uv`.
- Clarified the native-Windows TensorFlow GPU limitation.

### Version 1.0.1
- Fixed windowing behavior.

### Version 1.0.0
- Initial multi-framework support (nnU-Net V1/V2, PyTorch, Keras, MONAI).
- 2D/3D visualization and comparison mode.
- Preprocessing, Dice and IoU metrics, NIfTI export, STL/OBJ mesh export, DICOM support, and color customization.
