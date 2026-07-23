# SegMed - Medical Image Segmentation Tool

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

![SegMed LOGO](resource/imgs/Segmed.png)
> **SegMed: An Open-Source Desktop Tool for Deploying Pre-trained Deep Learning Models in 3D Medical Image Segmenta-tion**

A comprehensive medical image segmentation tool supporting multiple deep learning frameworks and providing advanced visualization capabilities for 3D medical imaging.


---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
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
- **Keras/TensorFlow Models**: Support for `.h5`, `.keras`, `.json`, and `.yaml` formats
- **MONAI Bundles**: Complete support for MONAI bundle architectures

### 🖼️ **Advanced Image Processing**
- **Multiple Image Formats**: DICOM, NIfTI (.nii, .nii.gz), NRRD (.nrrd, ..) and numpy bassed images (.npy)
- **DICOMDIR Support**: Load series from DICOMDIR files with series selection dialog
- **Window/Level Presets**: Pre-configured presets for brain, bone, lung, liver, and more
- **Custom Window/Level**: Manual adjustment for optimal visualization

### 🔧 **Preprocessing Pipeline**
- **Z-Normalization**: Image-based or manual (mean/std)
- **Min/Max Normalization**: Custom range normalization
- **Gaussian Blur**: Noise reduction with configurable sigma
- **Isotropic Resampling**: Uniform voxel spacing adjustment
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

### Prerequisites
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Step 1: Install uv

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone the Repository
```bash
git clone https://github.com/univpm-br3in/SegMed.git
cd SegMed
```

### Step 3: Run the main.py file
```bash
uv run main.py
```
On the first run, uv will create a virtual environment and install the dependencies specified in pyproject.toml. This may take several minutes. Subsequent runs will normally start much faster because the environment and downloaded packages are reused.

---
### 🚀 GPU Acceleration

SegMed includes both PyTorch and TensorFlow. By default, the project installs the standard versions specified in `pyproject.toml`. CPU inference is supported without additional configuration.

GPU acceleration is not configured automatically because the required packages depend on:

* The operating system
* The GPU manufacturer and model
* The installed GPU driver
* CUDA compatibility
* The selected deep-learning framework

#### PyTorch GPU Support

To use an NVIDIA GPU with PyTorch, install a CUDA-enabled build of `torch` and `torchvision` that is compatible with your system. Follow the official PyTorch installation instructions when selecting the appropriate CUDA version.

Verify the PyTorch configuration with:

```bash
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

#### TensorFlow GPU Support

TensorFlow GPU support depends on the operating system.

**Linux or Windows with WSL2**

For NVIDIA GPU support, install TensorFlow together with its CUDA dependencies:

```bash
uv add "tensorflow[and-cuda]>=2.20.0"
```

An appropriate NVIDIA driver must already be installed on the host system. The required CUDA runtime libraries are then installed in the project environment.

**Native Windows**

TensorFlow 2.10 was the last release that supported NVIDIA CUDA acceleration directly on native Windows. Because SegMed uses TensorFlow 2.20 or later, Windows users must run SegMed through WSL2 to use an NVIDIA GPU with TensorFlow.

Running TensorFlow 2.20 directly on native Windows will use the CPU.

**macOS**

CUDA acceleration is not available on macOS. Apple Silicon users may configure TensorFlow acceleration separately using Apple’s Metal plugin.

Verify whether TensorFlow can access a GPU with:

```bash
uv run python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

A correctly configured TensorFlow GPU environment should return at least one device in the `GPU devices` list.

> **Note:** PyTorch and TensorFlow use separate GPU configurations. Successfully enabling GPU acceleration for one framework does not necessarily mean that the other framework is also configured correctly. If no compatible GPU is detected, SegMed will fall back to CPU execution, which may result in slower inference.

---

## 🎮 Usage

### Basic Workflow

1. **Launch the Application**
   ```bash
   uv run main.py
   ```


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
   - Choose processing mode (CPU/GPU)
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
- **Saved Models**: `.h5`, `.keras` files
- **Architecture Files**: `.json` (with weights) or `.yaml` (with weights)
- **Custom Objects**: Automatic handling of custom layers/losses

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

**Note**: Preprocessing options can be combined and are applied in order.

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

## 📊 Performance Metrics

### DICE Coefficient (Sørensen-Dice Index)
Measures overlap similarity between segmentations:
```
DICE = 2 × |GT ∩ Pred| / (|GT| + |Pred|)
```
- Range: 0.0 (no overlap) to 1.0 (perfect match)
- More forgiving to size differences

### IoU (Intersection over Union / Jaccard Index)
Measures intersection relative to union:
```
IoU = |GT ∩ Pred| / |GT ∪ Pred|
```
- Range: 0.0 (no overlap) to 1.0 (perfect match)
- Stricter metric than DICE

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
├── pyproject.toml                   # uv/pip configuration
├── README.md                        # This file
└── LICENSE                          # License file
```

---

## 🎯 Tips & Best Practices

### Model Selection
- **nnU-Net**: Best for standard medical imaging tasks
- **MONAI**: Good for research and custom architectures
- **Custom PyTorch/Keras**: When you have specific requirements

### Preprocessing
- **nnU-Net Models**: Don't add extra preprocessing (already included)
- **Custom Models**: Apply preprocessing used during training
- **Isotropic Resampling**: Use for anisotropic scans (different spacing per axis)

### GPU Usage
- Enable GPU for faster inference.
- You need to install CUDA and related package depending on your system
- CPU mode works but is slower for large volumes
- nnU-Net benefits significantly from GPU acceleration

### Visualization
- Adjust window/level for better tissue contrast, or use None option.
- Use comparison mode to validate predictions
- Check all three orthogonal views for complete assessment

### Export
- Save segmentations immediately after successful inference
- Use multi-file export for separate class processing
- OBJ format is more compatible with 3D software

---

## 🐛 Troubleshooting

### Common Issues

**"No GPU available" warning**
- Install CUDA-compatible PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Or use CPU mode (slower but functional)

**Model loading fails**
- Check model format compatibility
- Ensure all required files are present (weights, config, etc.)
- For nnU-Net, verify folder structure

**Segmentation looks wrong**
- Verify correct preprocessing was applied
- Check if model expects specific input dimensions
- Confirm image orientation is correct

**3D visualization is slow**
- Reduce number of enabled classes
- Close other 3D applications
- Use simpler mesh export settings

**Metrics show zero**
- Ensure both GT and predicted segmentations are loaded
- Check that segmentations have matching dimensions
- Verify segmentation labels are > 0 (background is 0)

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
git clone https://github.com/yourusername/SegMed.git
cd SegMed
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"
```

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

### Version 1.0.0 (Current)
- ✅ Multi-framework model support (nnU-Net V1/V2, PyTorch, Keras, MONAI)
- ✅ 2D/3D visualization with comparison mode
- ✅ Comprehensive preprocessing pipeline
- ✅ DICE and IoU metrics calculation
- ✅ Mesh export (STL/OBJ)
- ✅ DICOM and NIfTI support
- ✅ Interactive color customization
