from PySide6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QRadioButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSlider,
    QGroupBox, QComboBox, QFileDialog,
    QDialog, QScrollArea
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QColor, QIcon
from PySide6.QtWidgets import QMessageBox

# Import PyQtGraph for 3D visualization
import pyqtgraph.opengl as gl
import numpy as np
import matplotlib.colors as mcolors
import torch
import tensorflow as tf
import nibabel as nib
import os
import sys
import scipy.ndimage as ndimage


from modules import images
from modules import mesh
from modules import networks
from modules.colorselectionwindow import ColorSelectionWindow
from modules.preprocessingwindow import PreProcessingWindow
from modules.preprocessingexecutor import PreprocessingExecutor
from modules.networkdimensioninputwindow import NetworkDimensionInputWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SegMed - Br3in")
        self.setMinimumSize(1200, 1000)
        self.setWindowIcon(QIcon(self.resource_path("resource/imgs/icon.ico")))

        # Initialize variables
        self.current_axial_slice = 0
        self.current_sagittal_slice = 0
        self.current_coronal_slice = 0
        self.enabeled_classes = []
        self.image = images.Image(-1, -1)  # ww, wl
        self.network = networks.Network()
        self.mesh = mesh.Mesh()
        self.mesh_items = []
        self.executor = PreprocessingExecutor()
        self.preprocssing_options = {}
        self.class_colors = []
        self.comparison_mode = False  # Flag for comparison display mode

        # Persistent color configuration
        self.saved_color_names = None
        self.saved_enabled_classes = None
        self.saved_class_names = None

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left panel for buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Group for load/save buttons
        operation_group = QGroupBox("Operations")
        operation_layout = QVBoxLayout(operation_group)

        # Load Network button
        self.load_network_btn = QPushButton("Load Network")
        self.load_network_btn.clicked.connect(self.load_network)
        operation_layout.addWidget(self.load_network_btn)

        # Load Image button
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        operation_layout.addWidget(self.load_image_btn)

        # Load segmentation button
        self.load_segmentation_btn = QPushButton("Load Segmentation")
        self.load_segmentation_btn.clicked.connect(self.load_segmentation)
        operation_layout.addWidget(self.load_segmentation_btn)

        # Export section
        export_widget = QWidget()
        export_layout = QVBoxLayout(export_widget)
        export_layout.setContentsMargins(0, 0, 0, 0)

        self.export_btn_seg = QPushButton("Export segmentation")
        self.export_btn_seg.clicked.connect(self.export_nii)
        export_layout.addWidget(self.export_btn_seg)

        self.export_btn = QPushButton("Export Mesh")
        self.export_btn.clicked.connect(self.export)
        export_layout.addWidget(self.export_btn)

        # export options
        export_options = QWidget()
        export_options_layout = QHBoxLayout(export_options)
        export_options_layout.setContentsMargins(20, 0, 0, 0)
        # First radio button group: File type (Single vs Multi)
        self.single_file_radio = QRadioButton("Single File")
        self.multi_files_radio = QRadioButton("Multi Files")
        self.single_file_radio.setChecked(True)
        export_options_layout.addWidget(self.single_file_radio)
        export_options_layout.addWidget(self.multi_files_radio)
        # Second radio button group: Format (STL vs OBJ)
        format_options = QWidget()
        format_options_layout = QHBoxLayout(format_options)
        format_options_layout.setContentsMargins(20, 0, 0, 0)
        self.stl_radio = QRadioButton("STL")
        self.obj_radio = QRadioButton("OBJ")
        self.stl_radio.setChecked(True)
        format_options_layout.addWidget(self.stl_radio)
        format_options_layout.addWidget(self.obj_radio)
        # Add both option groups to the layout
        export_layout.addWidget(export_options)
        export_layout.addWidget(format_options)
        operation_layout.addWidget(export_widget)

        # Segment button
        self.segment_btn = QPushButton("Segment!")
        self.segment_btn.clicked.connect(self.run_segmentation)
        operation_layout.addWidget(self.segment_btn)

        # Segment button
        self.preprocessing_btn = QPushButton("Preprocessing Options!")
        self.preprocessing_btn.clicked.connect(self.open_preprocessing_options)
        operation_layout.addWidget(self.preprocessing_btn)

        # Add operation group to left panel
        left_layout.addWidget(operation_group)

        # 2D image visulization options
        image_options_group = QGroupBox("2D Image Options")
        image_options_layout = QVBoxLayout(image_options_group)

        # Add a dropdown for selecting the window level presets
        preset_label = QLabel("Window Level Presets:")
        image_options_layout.addWidget(preset_label)
        self.wl_dropdown = QComboBox()
        self.wl_dropdown.addItems([
            "Costume", "Brain", "Subdural", "Stroke", "Lung", "Mediastinum", "Liver", "Abdomen", "Bone", "Angiography", "Heart", "None"
        ])
        self.wl_dropdown.setCurrentIndex(0)
        self.wl_dropdown.currentIndexChanged.connect(
            self.preset_update_window_level)

        left_layout.addWidget(image_options_group)
        image_options_layout.addWidget(self.wl_dropdown)
        # Add a label for the window level and width
        self.wl_label = QLabel(
            f"Level: {self.image.get_wl()}, Width: {self.image.get_ww()}")
        self.wl_label.setAlignment(Qt.AlignLeft)

        # Add a slider for window level
        self.w_label = QLabel("Manuel Level/Width:")
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setMinimum(-1000)
        self.wl_slider.setMaximum(1000)
        self.wl_slider.setValue(0)
        # Add a slider for window width
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setMinimum(1)
        self.ww_slider.setMaximum(2000)
        self.ww_slider.setValue(400)
        self.wl_slider.valueChanged.connect(self.preset_update_window_level)
        self.ww_slider.valueChanged.connect(self.preset_update_window_level)
        # Add a button to apply the window level and width
        apply_wl_btn = QPushButton("Apply")
        image_options_layout.addWidget(self.wl_label)
        image_options_layout.addWidget(self.w_label)
        image_options_layout.addWidget(self.wl_slider)
        image_options_layout.addWidget(self.ww_slider)
        image_options_layout.addWidget(apply_wl_btn)

        apply_wl_btn.clicked.connect(self.apply_window_level)

        # Processing mode radio buttons
        processing_group = QGroupBox("Processing Mode")
        processing_layout = QVBoxLayout(processing_group)

        self.gpu_radio = QRadioButton("GPU")
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setChecked(True)

        processing_layout.addWidget(self.gpu_radio)
        processing_layout.addWidget(self.cpu_radio)
        self.gpu_radio.toggled.connect(self.on_device_changed)
        self.cpu_radio.toggled.connect(self.on_device_changed)

        # Add processing group to left panel
        left_layout.addWidget(processing_group)

        # Segmentation mode radio buttons
        segmentation_group = QGroupBox("Segmentation:")
        segmentation_layout = QVBoxLayout(segmentation_group)

        self.gt_radio = QRadioButton("Loaded Segmentation")
        self.pre_radio = QRadioButton("Predicted Segmentation")
        self.compare_radio = QRadioButton("Compare Both")
        self.gt_radio.setEnabled(False)
        self.pre_radio.setEnabled(False)
        self.compare_radio.setEnabled(False)

        # Connect signals
        self.gt_radio.toggled.connect(self.on_segmentation_changed)
        self.pre_radio.toggled.connect(self.on_segmentation_changed)
        self.compare_radio.toggled.connect(self.on_segmentation_changed)

        segmentation_layout.addWidget(self.gt_radio)
        segmentation_layout.addWidget(self.pre_radio)
        segmentation_layout.addWidget(self.compare_radio)

        # Add processing group to left panel
        left_layout.addWidget(segmentation_group)

        # Metrics display group
        metrics_group = QGroupBox("Segmentation Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        # Create a scroll area for the metrics
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        metrics_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.metrics_label = QLabel(
            "No metrics available\n\nLoad both segmentations")
        self.metrics_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("""
            QLabel {
                padding: 5px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
            }
        """)

        # Add label to scroll area
        metrics_scroll.setWidget(self.metrics_label)
        metrics_scroll.setMinimumHeight(80)
        metrics_scroll.setMaximumHeight(120)

        metrics_layout.addWidget(metrics_scroll)

        left_layout.addWidget(metrics_group)

        # Exit button

        self.about_btn = QPushButton("About")
        self.about_btn.clicked.connect(self.show_about)
        left_layout.addWidget(self.about_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        left_layout.addWidget(self.exit_btn)

        # Add stretch to push everything up
        left_layout.addStretch()

        # Middle panel for image and controls - Modified for three orthogonal views
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create a container for the three views
        views_container = QWidget()
        views_layout = QGridLayout(views_container)

        # Axial view (top-left)
        axial_group = QGroupBox("Axial View")
        axial_layout = QVBoxLayout(axial_group)

        self.axial_label = QLabel("No image loaded")
        self.axial_label.setAlignment(Qt.AlignCenter)
        self.axial_label.setMinimumSize(300, 250)
        self.axial_label.setStyleSheet("background-color: #f0f0f0;")
        axial_layout.addWidget(self.axial_label)

        # Axial slider
        axial_slider_widget = QWidget()
        axial_slider_layout = QHBoxLayout(axial_slider_widget)
        axial_slider_layout.setContentsMargins(0, 0, 0, 0)

        axial_slider_label = QLabel("Slice:")
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(0)
        self.axial_slider.setValue(0)
        self.axial_slider.setEnabled(False)
        self.axial_slider.valueChanged.connect(
            lambda v: self.update_slice('axial', v))

        self.axial_value_label = QLabel("0/0")

        axial_slider_layout.addWidget(axial_slider_label)
        axial_slider_layout.addWidget(self.axial_slider)
        axial_slider_layout.addWidget(self.axial_value_label)

        axial_layout.addWidget(axial_slider_widget)

        # Sagittal view (top-right)
        sagittal_group = QGroupBox("Sagittal View")
        sagittal_layout = QVBoxLayout(sagittal_group)

        self.sagittal_label = QLabel("No image loaded")
        self.sagittal_label.setAlignment(Qt.AlignCenter)
        self.sagittal_label.setMinimumSize(300, 250)
        self.sagittal_label.setStyleSheet("background-color: #f0f0f0;")
        sagittal_layout.addWidget(self.sagittal_label)

        # Sagittal slider
        sagittal_slider_widget = QWidget()
        sagittal_slider_layout = QHBoxLayout(sagittal_slider_widget)
        sagittal_slider_layout.setContentsMargins(0, 0, 0, 0)

        sagittal_slider_label = QLabel("Slice:")
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setMinimum(0)
        self.sagittal_slider.setMaximum(0)
        self.sagittal_slider.setValue(0)
        self.sagittal_slider.setEnabled(False)
        self.sagittal_slider.valueChanged.connect(
            lambda v: self.update_slice('sagittal', v))

        self.sagittal_value_label = QLabel("0/0")

        sagittal_slider_layout.addWidget(sagittal_slider_label)
        sagittal_slider_layout.addWidget(self.sagittal_slider)
        sagittal_slider_layout.addWidget(self.sagittal_value_label)

        sagittal_layout.addWidget(sagittal_slider_widget)

        # Coronal view (bottom-left)
        coronal_group = QGroupBox("Coronal View")
        coronal_layout = QVBoxLayout(coronal_group)

        self.coronal_label = QLabel("No image loaded")
        self.coronal_label.setAlignment(Qt.AlignCenter)
        self.coronal_label.setMinimumSize(300, 250)
        self.coronal_label.setStyleSheet("background-color: #f0f0f0;")
        coronal_layout.addWidget(self.coronal_label)

        # Coronal slider
        coronal_slider_widget = QWidget()
        coronal_slider_layout = QHBoxLayout(coronal_slider_widget)
        coronal_slider_layout.setContentsMargins(0, 0, 0, 0)

        coronal_slider_label = QLabel("Slice:")
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setMinimum(0)
        self.coronal_slider.setMaximum(0)
        self.coronal_slider.setValue(0)
        self.coronal_slider.setEnabled(False)
        self.coronal_slider.valueChanged.connect(
            lambda v: self.update_slice('coronal', v))

        self.coronal_value_label = QLabel("0/0")

        coronal_slider_layout.addWidget(coronal_slider_label)
        coronal_slider_layout.addWidget(self.coronal_slider)
        coronal_slider_layout.addWidget(self.coronal_value_label)

        coronal_layout.addWidget(coronal_slider_widget)

        # 3D mesh view (bottom-right)
        mesh_group = QGroupBox("3D Mesh View")
        mesh_layout = QVBoxLayout(mesh_group)

        # Create a 3D plot widget using PyQtGraph
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setMinimumSize(300, 250)  # Match other views
        self.gl_widget.setCameraPosition(distance=40)

        # Add a grid to the 3D view for orientation
        grid = gl.GLGridItem()
        grid.setSize(20, 20, 1)
        grid.setSpacing(1, 1, 1)
        self.gl_widget.addItem(grid)

        mesh_layout.addWidget(self.gl_widget)

        # Change colors button
        self.change_colors_btn = QPushButton("Change Colors")
        self.change_colors_btn.clicked.connect(self.open_color_selection)
        mesh_layout.addWidget(self.change_colors_btn)

        # Add views to grid layout in 2x2 configuration
        views_layout.addWidget(axial_group, 0, 0)      # Top-left
        views_layout.addWidget(sagittal_group, 0, 1)   # Top-right
        views_layout.addWidget(coronal_group, 1, 0)    # Bottom-left
        views_layout.addWidget(mesh_group, 1, 1)       # Bottom-right

        # Add the views container to middle layout
        right_layout.addWidget(views_container)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)

        # Set the main widget
        self.setCentralWidget(main_widget)
        # Disable all buttons until an image is loaded
        self.load_image_btn.setEnabled(True)
        self.load_network_btn.setEnabled(True)
        self.segment_btn.setEnabled(False)
        self.export_btn.setEnabled(True)
        self.change_colors_btn.setEnabled(False)
        self.load_segmentation_btn.setEnabled(False)
        self.export_btn_seg.setEnabled(True)
        self.preset_update_window_level()

    def load_network(self):
        """Load neural network model with simplified YES/NO dialog."""
        # Ask user: single file or folder
        reply = QMessageBox.question(
            self,
            'Load Network',
            'Load single file or folder?\n\n'
            'YES - Single file (.ts, .pth, .pt, .h5, .keras, .json, .yaml)\n'
            'NO - Folder (MONAI Bundle or nnU-Net V2/V1)',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            # Load single file model
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Neural Network Model", "",
                "Model Files (*.ts *.json *.yaml *.yml *.pth *.h5 *.pt *.keras *.pt.tar *.pth.tar)"
            )
            if not file_path:
                return
            if self.gpu_radio.isChecked():
                if torch.cuda.is_available():
                    self.network.set_processing_type("gpu")
                else:
                    self.show_warning("GPU not available. Using CPU.")
                    self.network.set_processing_type("cpu")
                    self.cpu_radio.setChecked(True)
            else:
                self.network.set_processing_type("cpu")

            # Special handling for TorchScript files - ask if user wants to load weights
            if file_path.endswith('.ts'):
                weights_reply = QMessageBox.question(
                    self,
                    'TorchScript Model',
                    'This is a TorchScript (.ts) file.\n\n'
                    'Do you want to load weights from a separate .pth or .pt file?\n\n'
                    'YES - Select weights file (.pth, .pt)\n'
                    'NO - Use weights embedded in .ts file',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No  # Default to using embedded weights
                )

                if weights_reply == QMessageBox.Yes:
                    # Ask for weights file
                    weights_path, _ = QFileDialog.getOpenFileName(
                        self, "Select Weights File", "",
                        "PyTorch Weights (*.pth *.pt)"
                    )
                    if weights_path:
                        # Load model with separate weights
                        if self.network.load_model(file_path, weights_path=weights_path):
                            self.segment_btn.setEnabled(True)
                            self.show_warning(
                                f"TorchScript model loaded successfully with external weights!\n"
                                f"Model: {os.path.basename(file_path)}\n"
                                f"Weights: {os.path.basename(weights_path)}\n"
                                f"Input dimensions: {self.network.input_dims}")
                        else:
                            self.show_warning(
                                "Failed to load model with external weights.")
                    return

            # Standard loading for all other files
            if self.network.load_model(file_path):
                self.segment_btn.setEnabled(True)
                model_type = self.network.model_type or "Unknown"
                self.show_warning(
                    f"Model loaded successfully!\n"
                    f"Type: {model_type}\n"
                    f"Input dimensions: {self.network.input_dims}")
            else:
                self.show_warning(
                    "Model was not loaded. Please make sure the file is a valid model format.")

        else:  # User selected NO - load folder
            # Load folder (could be MONAI or nnU-Net)
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Model Directory (MONAI Bundle or nnU-Net V1/V2)", ""
            )
            if not dir_path:
                return
            if self.gpu_radio.isChecked():
                if torch.cuda.is_available():
                    self.network.set_processing_type("gpu")
                else:
                    self.show_warning("GPU not available. Using CPU.")
                    self.network.set_processing_type("cpu")
                    self.cpu_radio.setChecked(True)
            else:
                self.network.set_processing_type("cpu")
            # Auto-detect if it's MONAI or nnU-Net (returns tuple: (is_nnunet, version))
            is_nnunet, nnunet_version = self.network._is_nnunet_folder(
                dir_path)
            is_monai = self.network._is_monai_bundle(dir_path)

            if is_nnunet:
                # Load as nnU-Net (V1 or V2 auto-detected)
                if self.network._load_nnunet_model(dir_path):
                    self.segment_btn.setEnabled(True)
                    info = self.network.get_nnunet_class_info()
                    if info:
                        version_str = info.get('version', f'V{nnunet_version}')
                        self.show_warning(
                            f"nnU-Net {version_str} Model loaded successfully!\n"
                            f"Configuration: {info['configuration']}\n"
                            f"Input dimensions: {self.network.input_dims}\n"
                            f"Number of classes: {info['num_classes']}\n"
                            f"Patch size: {info['patch_size']}")
                    else:
                        self.show_warning(
                            f"nnU-Net V{nnunet_version} Model loaded successfully!")
                else:
                    self.show_warning(
                        "Failed to load nnU-Net model. Please check the model directory.")

            elif is_monai:
                # Load as MONAI bundle
                if self.network._load_monai_bundle(dir_path):
                    self.segment_btn.setEnabled(True)
                    self.show_warning(
                        f"MONAI Bundle loaded successfully!\n"
                        f"Input dimensions: {self.network.input_dims}")
                else:
                    self.show_warning(
                        "Failed to load MONAI bundle. Please check the bundle directory.")

            else:
                # Unknown folder structure
                self.show_warning(
                    "Could not detect model type in selected directory.\n\n"
                    "Expected:\n"
                    "- MONAI Bundle: configs/ and models/ directories\n"
                    "- nnU-Net V2: plans.json and fold_X/ directories\n"
                    "- nnU-Net V1: plans.pkl and fold_X/ directories\n\n"
                    "Please select a valid model directory.")

    def load_image(self):
        # Ask user what type of selection they want
        reply = QMessageBox.question(
            self,
            'Load Image',
            'Load single file or DICOM series?\n\n'
            'YES - Single file (.nii, .nii.gz, .npy, single .dcm, , .nrrd, .nhdr, dicomdir)\n'
            'NO - DICOM series (directory)',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        path = None
        if reply == QMessageBox.Yes:
            # Load single file
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Image File", "",
                "Image Files (*.nii *.nii.gz *.dcm *.npy *.nrrd *.nhdr dicomdir);;NIfTI Files (*.nii *.nii.gz);;DICOM Files (*.dcm);;NumPy Files (*.npy);;NRRD Files (*.nrrd *.nhdr);; DICOMDIR (dicomdir)"
            )
        else:
            # Load directory (for DICOM series)
            path = QFileDialog.getExistingDirectory(
                self, "Select DICOM Series Directory", ""
            )

        if path:
            info = self.image.load_image(path)
            if info is None:
                # Clear cached meshes when loading new image
                self.invalidate_cached_meshes()

                # Update display
                self.update_all_sliders()
                self.load_segmentation_btn.setEnabled(True)
                self.image.GT_segmentation = None
                self.image.pre_segmentation = None
                self.mesh.set_spacing(self.image.get_image_spacing())
                self.mesh.set_origin(self.image.get_origin())
                self.mesh.set_direction(self.image.get_direction())

                for item in self.mesh_items:
                    if item in self.gl_widget.items:
                        self.gl_widget.removeItem(item)
                self.mesh_items = []

                self.gt_radio.setEnabled(False)
                self.pre_radio.setEnabled(False)
                self.gt_radio.setChecked(False)
                self.pre_radio.setChecked(False)
            else:
                self.show_warning(
                    f"Image is not in 3D, size is: {info}, use 3D images only, image was not loaded")

    def load_segmentation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Segmentation", "", "Image Files (*.nii *.nii.gz *.dcm *.npy *.nrrd *.nhdr);;NIfTI Files (*.nii *.nii.gz);;DICOM Files (*.dcm);;NumPy Files (*.npy);;NRRD Files (*.nrrd *.nhdr)"
        )
        if file_path:
            info = self.image.load_seg(file_path)
            if not info:
                self.show_warning(
                    f"Segmentation shape does not match image shape!, segmentation was not loaded")
                return

            self.mesh.create_faces_triangles_GT(
                self.image.get_GT_segmentation())
            self.open_color_selection()
            self.update_all_sliders()

            # Create cached GT meshes
            self.create_gt_meshes()
            self.mesh_items = self.gt_mesh_items

            self.color_mesh()
            self.change_colors_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            self.gt_radio.setEnabled(True)
            self.gt_radio.setChecked(True)

            # Invalidate metrics cache since we have new GT
            if hasattr(self, '_cached_metrics'):
                delattr(self, '_cached_metrics')

            # Enable comparison mode if both GT and predicted segmentations exist
            if self.image.get_GT_segmentation() is not None and self.image.get_pre_segmentation() is not None:
                self.compare_radio.setEnabled(True)

            # Update metrics display
            self.update_metrics_display()

    def run_segmentation(self):
        # First check if a model and image are loaded
        if self.network.model is None:
            self.show_warning("No neural network model loaded!")
            return
        if self.image.get_processed_image() is None:
            self.show_warning("No image loaded!")
            return

        # Determine processing mode
        if self.gpu_radio.isChecked():
            if (torch.cuda.is_available() or tf.config.list_physical_devices('GPU')):
                self.network.set_processing_type("gpu")
            else:
                self.show_warning("No GPU available, switching to CPU mode.")
                self.network.set_processing_type("cpu")
                self.cpu_radio.setChecked(True)
        else:
            self.network.set_processing_type("cpu")
            self.cpu_radio.setChecked(True)

        if self.network.model_type == "monai":
            A = self.network.get_prediction(self.image.get_image_data(),
                                            spacing=self.image.get_image_spacing(),
                                            origin=self.image.get_origin(),
                                            direction=self.image.get_direction())
            self.image.set_pre_segmentation(A, "3D")

        elif self.network.model_type == "nnunet":
            A = self.network.get_prediction(self.image.get_image_data(),
                                            spacing=self.image.get_image_spacing(),
                                            origin=self.image.get_origin(),
                                            direction=self.image.get_direction())
            self.image.set_pre_segmentation(A, "3D")

        else:
            # Provide missing preprocessing values
            if self.preprocssing_options:
                if self.preprocssing_options['Isotropic Resampling']["enabled"]:
                    self.preprocssing_options['Isotropic Resampling']["values"] = self.image.get_image_spacing(
                    )

            # Check if the dimensions of the image are compatible with the model
            is_compatible, neededSize, mode = self.network.check_model_compatibility(
                self.image.get_image_shape())

            if not is_compatible and neededSize is not None:
                self.show_warning(
                    f"Image dimensions {self.image.get_image_shape()} are not compatible with the model, image will be {mode}!")

                resized_image = self.image.resize_image(
                    neededSize, self.network.input_dims, mode)
                if "-" in mode:
                    mode = mode.split("-")[1]

                if self.preprocssing_options:
                    resized_image = self.executor.execute_preprocessing(
                        resized_image, self.preprocssing_options)

                self.image.set_pre_segmentation(
                    self.image.restore_size(self.network.get_prediction(resized_image).squeeze(), mode, self.image.get_image_shape()), self.network.input_dims)

            elif not is_compatible and neededSize is None:
                # Open manual input dialog
                dialog = NetworkDimensionInputWindow(self)
                dialog.dimensions_configured.connect(
                    lambda dims, channel, mode: self.process_with_manual_dimensions(
                        dims, channel, mode)
                )

                result = dialog.exec()
                if result == QDialog.Accepted:
                    # Processing will be handled by the signal callback
                    pass
                else:
                    # User cancelled
                    self.show_warning(
                        "Segmentation cancelled. Please provide network dimensions to proceed.")
                return
            else:
                if self.preprocssing_options:
                    processed_image = self.executor.execute_preprocessing(
                        self.image.get_processed_image().squeeze(), self.preprocssing_options)
                    print(processed_image.max())
                else:
                    processed_image = self.image.get_processed_image()

                self.image.set_pre_segmentation(
                    self.network.get_prediction(processed_image), self.network.input_dims)

        self.mesh.create_faces_triangles_pre(self.image.get_pre_segmentation())
        self.open_color_selection()

        # Create cached predicted meshes
        self.create_pre_meshes()
        self.mesh_items = self.pre_mesh_items

        self.color_mesh()

        self.pre_radio.setChecked(True)
        self.pre_radio.setEnabled(True)
        self.change_colors_btn.setEnabled(True)

        # Invalidate metrics cache since we have new prediction
        if hasattr(self, '_cached_metrics'):
            delattr(self, '_cached_metrics')

        # Enable comparison mode if both GT and predicted segmentations exist
        if self.image.get_GT_segmentation() is not None and self.image.get_pre_segmentation() is not None:
            self.compare_radio.setEnabled(True)

        # Update metrics display
        self.update_metrics_display()

    def process_with_manual_dimensions(self, dimensions, channel_format, processing_mode):
        """
        Process segmentation with manually provided network dimensions.

        Args:
            dimensions (tuple): The expected input dimensions (width, height) or (width, height, depth)
            channel_format (str): 'channel_first', 'channel_last', or 'none'
            processing_mode (str): 'pad' or 'resize'
        """
        try:
            # Determine network dimensionality from dimensions tuple
            network_dims = "3D" if len(dimensions) == 3 else "2D"

            # Get the image
            image_data = self.image.get_processed_image().squeeze()

            # Apply preprocessing if configured
            if self.preprocssing_options:
                if self.preprocssing_options['Isotropic Resampling']["enabled"]:
                    self.preprocssing_options['Isotropic Resampling']["values"] = self.image.get_image_spacing(
                    )

                image_data = self.executor.execute_preprocessing(
                    image_data, self.preprocssing_options)

            # Resize or pad the image based on the processing mode
            mode_string = f"{channel_format}-{processing_mode}"
            resized_image = self.image.resize_image(
                dimensions, network_dims, mode_string)

            # Get prediction
            prediction = self.network.get_prediction(resized_image).squeeze()

            # Restore original size
            restored_prediction = self.image.restore_size(
                prediction, processing_mode, self.image.get_image_shape()
            )

            # Set the prediction
            self.image.set_pre_segmentation(restored_prediction, network_dims)

            # Create meshes and visualizations
            self.mesh.create_faces_triangles_pre(
                self.image.get_pre_segmentation())
            self.open_color_selection()

            # Create cached predicted meshes
            self.create_pre_meshes()
            self.mesh_items = self.pre_mesh_items

            self.color_mesh()

            # Enable predicted segmentation radio button
            self.pre_radio.setChecked(True)
            self.pre_radio.setEnabled(True)
            self.change_colors_btn.setEnabled(True)

            QMessageBox.information(
                self, "Success",
                "Segmentation completed successfully with manual dimensions!"
            )

        except Exception as e:
            self.show_warning(
                f"Error during segmentation with manual dimensions: {str(e)}")
            import traceback
            traceback.print_exc()

    def export(self):
        if self.gt_radio.isChecked():
            gtFlag = True
        else:
            gtFlag = False

        if self.stl_radio.isChecked():
            self.export_stl(gtFlag)
        elif self.obj_radio.isChecked():
            self.export_obj(gtFlag)

    def export_stl(self, gtFlag):
        if not self.mesh_items:
            print("No mesh to export")
            return

        if self.single_file_radio.isChecked():
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export STL", "", "STL Files (*.stl)"
            )
            if file_path:
                self.mesh.export_stl(file_path, gtFlag, self.enabeled_classes)
        else:
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Directory for STL Files"
            )
            if dir_path:
                self.mesh.export_stl_files(
                    dir_path, gtFlag, self.enabeled_classes, self.saved_class_names)

    def export_obj(self, gtFlag):
        if not self.mesh_items:
            print("No mesh to export")
            return

        if self.single_file_radio.isChecked():
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export OBJ", "", "OBJ Files (*.OBJ)"
            )
            if file_path:
                self.mesh.export_obj(file_path, gtFlag, self.enabeled_classes)
        else:
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Directory for OBJ Files"
            )
            if dir_path:
                self.mesh.export_obj_files(
                    dir_path, gtFlag, self.enabeled_classes, self.saved_class_names)

    def export_nii(self):
        if self.image.get_pre_segmentation() is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Segmentation", "", "NIfTI Files (*.nii *.nii.gz)"
        )

        if file_path:
            affine = np.eye(4)
            affine[:3, :3] = self.image.get_direction() * \
                self.image.get_image_spacing()
            affine[:3, 3] = self.image.get_origin()
            nii_img = nib.Nifti1Image(
                self.image.get_pre_segmentation(), affine)
            if not file_path.endswith(('.nii', '.nii.gz')):
                file_path += '.nii.gz'
            nib.save(nii_img, file_path)

    def open_color_selection(self):
        numOfClasses = self.image.get_num_of_classes()
        dialog = ColorSelectionWindow(
            self,
            numOfClasses-1,
            initial_colors=self.saved_color_names,
            initial_enabled=self.saved_enabled_classes,
            initial_names=self.saved_class_names
        )
        result = dialog.exec()
        if result:
            colors = [widget[1].currentText()
                      for widget in dialog.class_widgets]
            self.class_colors = [
                tuple(int(255 * c) for c in mcolors.to_rgb(color)) for color in colors]
            self.enabeled_classes = [widget[0].isChecked()
                                     for widget in dialog.class_widgets]
            self.mesh_colors = [QColor(color) for color in colors]
            self.class_enabled = [enable for enable in self.enabeled_classes]

            # Save configuration for next time
            self.saved_color_names = colors
            self.saved_enabled_classes = self.enabeled_classes.copy()
            self.saved_class_names = dialog.get_class_names()

            self.update_all_sliders()
        if self.mesh_items:
            self.color_mesh()

    def preset_update_window_level(self):
        """
        Update the window level based on the selected preset.
        """
        preset = self.wl_dropdown.currentText()
        # CT Window/Level Presets (WL, WW)
        wl_ww_presets = {
            "brain": (40, 80),
            "subdural": (50, 130),
            # very narrow window for ischemia detection
            "stroke": (35, 25),
            "lung": (-600, 1500),
            "mediastinum": (40, 400),  # general chest/soft tissue
            "liver": (50, 150),
            "abdomen": (60, 400),
            "bone": (400, 1800),       # sometimes wider: (300, 2000)
            "angiography": (300, 600),
            # similar to mediastinum, highlights myocardium & coronary arteries
            "heart": (40, 350),
            "none": (-1, -1)
        }
        if preset.lower() == "costume":
            self.wl_slider.setEnabled(True)
            self.ww_slider.setEnabled(True)
            self.wl = self.wl_slider.value()
            self.ww = self.ww_slider.value()
        else:
            self.wl_slider.setEnabled(False)
            self.ww_slider.setEnabled(False)
            self.wl, self.ww = wl_ww_presets[preset.lower()]

        self.wl_label.setText(f"Level: {self.wl}, Width: {self.ww}")

    def apply_window_level(self):
        """
        Apply the window level and width to the image.
        """
        # Update the image display
        if self.image.get_processed_image() is not None:
            self.image.set_window_level(self.wl, self.ww)
            self.update_all_sliders()

    def show_warning(self, message):
        """
        Show a warning message box.
        :param message: Warning message to display.
        """
        QMessageBox.warning(
            self, "Warning", message, QMessageBox.Ok
        )

    def show_about(self):
        """
        Open the about dialog that shows the br3in.png and univpm.png logos.
        """
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About SegMed")
        about_dialog.setFixedSize(250, 350)

        main_layout = QVBoxLayout(about_dialog)

        # Segmed logo
        Segmed_logo_label = QLabel()
        Segmed_pixmap = QPixmap(self.resource_path("resource/imgs/Segmed.png"))
        Segmed_logo_label.setPixmap(Segmed_pixmap.scaled(
            QSize(100, 100), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        Segmed_logo_label.setAlignment(Qt.AlignCenter)

        # Horizontal layout for logos
        logos_layout = QHBoxLayout()

        # Br3in logo
        br3in_logo_label = QLabel()
        br3in_pixmap = QPixmap(self.resource_path("resource/imgs/univpm.png"))
        br3in_logo_label.setPixmap(br3in_pixmap.scaled(
            QSize(100, 100), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        br3in_logo_label.setAlignment(Qt.AlignLeft)

        # Univpm logo
        univpm_logo_label = QLabel()
        univpm_pixmap = QPixmap(self.resource_path("resource/imgs/br3in.png"))
        univpm_logo_label.setPixmap(univpm_pixmap.scaled(
            QSize(100, 100), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        univpm_logo_label.setAlignment(Qt.AlignRight)

        # Add logos to horizontal layout
        logos_layout.addWidget(br3in_logo_label)
        logos_layout.addWidget(univpm_logo_label)

        # Info text
        info_label = QLabel(
            "SegMed - Br3in\nDeveloped at \nUniversità Politecnica delle Marche\nhttps://br3in.dii.univpm.it/")
        info_label.setAlignment(Qt.AlignCenter)

        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(about_dialog.accept)

        # Add widgets to main layout
        main_layout.addWidget(Segmed_logo_label)
        main_layout.addLayout(logos_layout)
        main_layout.addWidget(info_label)
        main_layout.addWidget(ok_button, alignment=Qt.AlignCenter)

        about_dialog.exec()

    def update_slice(self, view_type, slice_index):
        """Update the slice for the specified view type (axial, sagittal, or coronal)"""
        if self.image.get_processed_image() is None:
            return

        # Update the current slice for the specific view
        if view_type == 'axial':
            self.current_axial_slice = slice_index
        elif view_type == 'sagittal':
            self.current_sagittal_slice = slice_index
        elif view_type == 'coronal':
            self.current_coronal_slice = slice_index

        # Get the appropriate image slice based on view type
        image_data = self.image.get_processed_image()
        spacing = self.image.get_image_spacing()
        if image_data.max() == 255.0:
            norlimalized = False
        else:
            norlimalized = True
            max_val = image_data.max()
            min_val = image_data.min()

        if view_type == 'axial':
            # Axial view - slice along z-axis (depth)
            slice_data = image_data[:, :, slice_index]
            slice_data = self.rescale_slice(
                slice_data, (spacing[0], spacing[1]))
            image_label = self.axial_label
            value_label = self.axial_value_label
            slider = self.axial_slider
        elif view_type == 'sagittal':
            # Sagittal view - slice along x-axis (left-right)
            slice_data = image_data[slice_index, :, :]
            slice_data = self.rescale_slice(
                slice_data, (spacing[1], spacing[2]))
            # Apply rotation only if needed for proper medical display
            if self.image.get_sagittal_rotation_needed():
                slice_data = np.rot90(slice_data)
            image_label = self.sagittal_label
            value_label = self.sagittal_value_label
            slider = self.sagittal_slider
        elif view_type == 'coronal':
            # Coronal view - slice along y-axis (front-back)
            slice_data = image_data[:, slice_index, :]
            slice_data = self.rescale_slice(
                slice_data, (spacing[0], spacing[2]))
            # Apply rotation only if needed for proper medical display
            if self.image.get_coronal_rotation_needed():
                slice_data = np.rot90(slice_data)
            image_label = self.coronal_label
            value_label = self.coronal_value_label
            slider = self.coronal_slider
        if norlimalized:
            slice_data = ((slice_data - min_val) / (max_val - min_val))
            slice_data = (slice_data * 255).clip(0, 255).astype(np.uint8)

        slice_data = np.ascontiguousarray(slice_data)
        show_seg = False
        show_comparison = False

        # Check if segmentation should be shown
        if self.gt_radio.isChecked() and self.image.get_GT_segmentation() is not None:
            show_seg = True
            seg_data = self.image.get_GT_segmentation()
        elif self.pre_radio.isChecked() and self.image.get_pre_segmentation() is not None:
            show_seg = True
            seg_data = self.image.get_pre_segmentation()
        elif self.compare_radio.isChecked() and self.image.get_GT_segmentation() is not None and self.image.get_pre_segmentation() is not None:
            show_comparison = True
            gt_seg_data = self.image.get_GT_segmentation()
            pre_seg_data = self.image.get_pre_segmentation()

        # Create the image with or without segmentation overlay
        if show_comparison:
            # COMPARISON MODE: Show predicted as solid + GT as contour
            # Get both segmentation slices based on view type
            if view_type == 'axial':
                gt_seg_slice = gt_seg_data[:, :, slice_index]
                gt_seg_slice = self.rescale_slice(
                    gt_seg_slice, (spacing[0], spacing[1]), 0)
                pre_seg_slice = pre_seg_data[:, :, slice_index]
                pre_seg_slice = self.rescale_slice(
                    pre_seg_slice, (spacing[0], spacing[1]), 0)
            elif view_type == 'sagittal':
                gt_seg_slice = gt_seg_data[slice_index, :, :]
                gt_seg_slice = self.rescale_slice(
                    gt_seg_slice, (spacing[1], spacing[2]), 0)
                pre_seg_slice = pre_seg_data[slice_index, :, :]
                pre_seg_slice = self.rescale_slice(
                    pre_seg_slice, (spacing[1], spacing[2]), 0)
                if self.image.get_sagittal_rotation_needed():
                    gt_seg_slice = np.rot90(gt_seg_slice)
                    pre_seg_slice = np.rot90(pre_seg_slice)
            elif view_type == 'coronal':
                gt_seg_slice = gt_seg_data[:, slice_index, :]
                gt_seg_slice = self.rescale_slice(
                    gt_seg_slice, (spacing[0], spacing[2]), 0)
                pre_seg_slice = pre_seg_data[:, slice_index, :]
                pre_seg_slice = self.rescale_slice(
                    pre_seg_slice, (spacing[0], spacing[2]), 0)
                if self.image.get_coronal_rotation_needed():
                    gt_seg_slice = np.rot90(gt_seg_slice)
                    pre_seg_slice = np.rot90(pre_seg_slice)

            # Create an RGB image starting with the original image data
            height, width = pre_seg_slice.shape
            rgb_image = np.stack((slice_data,) * 3, axis=-1)

            # First, draw predicted segmentation as solid overlay
            for class_id in range(1, self.image.get_num_of_classes()):
                if class_id - 1 >= len(self.enabeled_classes) or not self.enabeled_classes[class_id-1]:
                    continue
                color = self.class_colors[class_id-1]
                mask = pre_seg_slice == class_id
                if np.any(mask):
                    alpha = 0.4  # Semi-transparent for predicted
                    rgb_image[mask] = alpha * \
                        np.array(color) + (1 - alpha) * rgb_image[mask]

            # Then, draw GT segmentation as contours
            from scipy import ndimage
            for class_id in range(1, self.image.get_num_of_classes()):
                if class_id - 1 >= len(self.enabeled_classes) or not self.enabeled_classes[class_id-1]:
                    continue
                color = self.class_colors[class_id-1]
                mask = gt_seg_slice == class_id
                if np.any(mask):
                    # Find contours using edge detection
                    edges = ndimage.binary_dilation(mask) & ~mask
                    # Make contours thicker for visibility
                    edges = ndimage.binary_dilation(edges, iterations=2)
                    # Draw bright contour
                    rgb_image[edges] = color

            # Convert to QImage
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            rgb_image = rgb_image.astype(np.uint8)
            rgb_image = np.ascontiguousarray(rgb_image)
            q_image = QImage(rgb_image.data, width, height,
                             bytes_per_line, QImage.Format_RGB888)

        elif show_seg:
            # NORMAL MODE: Single segmentation overlay
            # Get segmentation slice based on view type
            if view_type == 'axial':
                segmentation_slice = seg_data[:, :, slice_index]
                segmentation_slice = self.rescale_slice(
                    segmentation_slice, (spacing[0], spacing[1]), 0)
            elif view_type == 'sagittal':
                segmentation_slice = seg_data[slice_index, :, :]
                segmentation_slice = self.rescale_slice(
                    segmentation_slice, (spacing[1], spacing[2]), 0)
                # Apply rotation only if needed for proper medical display
                if self.image.get_sagittal_rotation_needed():
                    segmentation_slice = np.rot90(segmentation_slice)
            elif view_type == 'coronal':
                segmentation_slice = seg_data[:, slice_index, :]
                segmentation_slice = self.rescale_slice(
                    segmentation_slice, (spacing[0], spacing[2]), 0)
                # Apply rotation only if needed for proper medical display
                if self.image.get_coronal_rotation_needed():
                    segmentation_slice = np.rot90(segmentation_slice)

            # Create an RGB image starting with the original image data
            height, width = segmentation_slice.shape

            # Convert grayscale image to RGB by stacking it 3 times
            rgb_image = np.stack((slice_data,) * 3, axis=-1)

            # For each class in the segmentation, apply the class color
            for class_id in range(1, self.image.get_num_of_classes()):
                color = self.class_colors[class_id-1]
                # Create mask for this class
                mask = segmentation_slice == class_id
                # Check if this class exists in the slice
                if np.any(mask) and self.enabeled_classes[class_id-1]:
                    # Apply alpha blending - overlay color on original image
                    alpha = 0.5  # Transparency level
                    rgb_image[mask] = alpha * \
                        np.array(color) + (1 - alpha) * rgb_image[mask]

            # Convert numpy array to QImage
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            # Convert to uint8 if not already
            rgb_image = rgb_image.astype(np.uint8)
            rgb_image = np.ascontiguousarray(rgb_image)

            q_image = QImage(rgb_image.data, width, height,
                             bytes_per_line, QImage.Format_RGB888)
        else:
            # Convert numpy array to QImage (grayscale)
            height, width = slice_data.shape
            bytes_per_line = width
            q_image = QImage(slice_data.data, width, height,
                             bytes_per_line, QImage.Format_Grayscale8)
        # Create pixmap and scale it
        pixmap = QPixmap.fromImage(q_image)

        # Scale the pixmap to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            image_label.width(),
            image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Set the pixmap to the appropriate label
        image_label.setPixmap(pixmap)

        # Update the slice value label
        value_label.setText(f"{slice_index}/{slider.maximum()}")

    def update_all_sliders(self):
        """Update all slider ranges when a new image is loaded"""
        if self.image.get_processed_image() is not None:
            shape = self.image.get_image_shape()

            # Axial slider (z-axis)
            self.axial_slider.setMaximum(shape[2] - 1)
            self.axial_slider.setEnabled(True)
            self.axial_slider.setValue(shape[2] // 2)  # Start at middle slice

            # Sagittal slider (x-axis)
            self.sagittal_slider.setMaximum(shape[0] - 1)
            self.sagittal_slider.setEnabled(True)
            self.sagittal_slider.setValue(
                shape[0] // 2)  # Start at middle slice

            # Coronal slider (y-axis)
            self.coronal_slider.setMaximum(shape[1] - 1)
            self.coronal_slider.setEnabled(True)
            self.coronal_slider.setValue(
                shape[1] // 2)  # Start at middle slice

            # Update all views
            self.update_slice('axial', self.axial_slider.value())
            self.update_slice('sagittal', self.sagittal_slider.value())
            self.update_slice('coronal', self.coronal_slider.value())

    def update_current_slices(self):
        """Update slice views at current positions without resetting slider values.

        This is used when switching between segmentation modes (GT/Predicted/Compare)
        to refresh the display while maintaining the user's current view position.
        """
        if self.image.get_processed_image() is not None:
            # Just update the views at current slider positions
            self.update_slice('axial', self.axial_slider.value())
            self.update_slice('sagittal', self.sagittal_slider.value())
            self.update_slice('coronal', self.coronal_slider.value())

    def update_metrics_display(self):
        """Update the metrics label with current DICE and IOU scores."""
        if self.image.get_GT_segmentation() is None or self.image.get_pre_segmentation() is None:
            self.metrics_label.setText(
                "No metrics available\n\nLoad both segmentations")
            return

        # Calculate metrics if not cached
        if not hasattr(self, '_cached_metrics'):
            gt_full = self.image.get_GT_segmentation()
            pred_full = self.image.get_pre_segmentation()
            self._cached_metrics = self.calculate_metrics(gt_full, pred_full)

        # Format metrics text - compact version
        text_lines = []

        # Add overall metrics first
        if 'Overall' in self._cached_metrics:
            text_lines.append("OVERALL:")
            text_lines.append(
                f"DICE: {self._cached_metrics['Overall']['dice']:.4f}")
            text_lines.append(
                f"IOU:  {self._cached_metrics['Overall']['iou']:.4f}")

        # Add per-class metrics
        for class_name, metrics in self._cached_metrics.items():
            if class_name != 'Overall':
                text_lines.append(f"\n{class_name}:")
                text_lines.append(
                    f"D:{metrics['dice']:.3f} I:{metrics['iou']:.3f}")

        self.metrics_label.setText("\n".join(text_lines))

    def calculate_metrics(self, gt_seg, pred_seg):
        """
        Calculate DICE and IOU metrics for each class.

        Args:
            gt_seg: Ground truth segmentation
            pred_seg: Predicted segmentation

        Returns:
            dict: Dictionary with class-wise and overall metrics
        """
        metrics = {}

        # Get unique classes (excluding background)
        classes = np.unique(np.concatenate(
            [gt_seg.flatten(), pred_seg.flatten()]))
        classes = classes[classes > 0]  # Remove background

        dice_scores = []
        iou_scores = []

        for class_id in classes:
            # Create binary masks for this class
            gt_mask = (gt_seg == class_id).astype(np.float32)
            pred_mask = (pred_seg == class_id).astype(np.float32)

            # Calculate intersection and union
            intersection = np.sum(gt_mask * pred_mask)
            gt_sum = np.sum(gt_mask)
            pred_sum = np.sum(pred_mask)
            union = gt_sum + pred_sum - intersection

            # Calculate DICE coefficient
            if gt_sum + pred_sum > 0:
                dice = (2.0 * intersection) / (gt_sum + pred_sum)
            else:
                dice = 1.0 if intersection == 0 else 0.0

            # Calculate IOU (Jaccard Index)
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if intersection == 0 else 0.0

            dice_scores.append(dice)
            iou_scores.append(iou)

            # Store per-class metrics
            class_name = self.saved_class_names[int(class_id)-1] if self.saved_class_names and int(
                class_id)-1 < len(self.saved_class_names) else f"Class {int(class_id)}"
            metrics[class_name] = {
                'dice': dice,
                'iou': iou
            }

        # Calculate overall metrics (mean across classes)
        if dice_scores:
            metrics['Overall'] = {
                'dice': np.mean(dice_scores),
                'iou': np.mean(iou_scores)
            }
        else:
            metrics['Overall'] = {
                'dice': 0.0,
                'iou': 0.0
            }

        return metrics

    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        if hasattr(sys, '_MEIPASS'):
            # Running in exe
            base_path = sys._MEIPASS
        else:
            # Running in dev
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def rescale_slice(self, slice_data, spacing, order=1):
        """Rescale slice based on voxel spacing"""
        zoom_factors = (
            spacing[0] / min(spacing),  # x scaling
            spacing[1] / min(spacing)   # y scaling
        )
        return ndimage.zoom(slice_data, zoom_factors, order=order)  # bilinear

    def open_preprocessing_options(self):
        """Open preprocessing options dialog"""
        if self.image.get_processed_image() is None:
            return
        dialog = PreProcessingWindow(self)
        if dialog.exec() == QDialog.Accepted:
            self.preprocssing_options = dialog.get_selected_options()
        if self.preprocssing_options:
            if self.preprocssing_options['Isotropic Resampling']["enabled"]:
                self.preprocssing_options['Isotropic Resampling']["values"] = self.image.get_image_spacing(
                )

    def create_gt_meshes(self):
        """Create and cache GT (Ground Truth) mesh items"""
        if not hasattr(self, 'gt_mesh_items'):
            self.gt_mesh_items = []

        # Clear existing GT meshes
        for item in self.gt_mesh_items:
            if item in self.gl_widget.items:
                self.gl_widget.removeItem(item)
        self.gt_mesh_items.clear()

        if self.image.get_GT_segmentation() is None:
            return

        # Get GT mesh data
        vertices = self.mesh.get_verticesGT()
        faces = self.mesh.get_facesGT()
        faces_class = self.mesh.get_classesGT()

        if vertices is None or faces is None or faces_class is None:
            return

        # FIXED: Get actual unique classes from the segmentation, not from num_classes
        segmentation = self.image.get_GT_segmentation()
        actual_unique_classes = np.unique(segmentation)
        # Filter out background (0) and convert to list
        unique_classes = [int(c) for c in actual_unique_classes if c > 0]

        vertices = vertices - np.mean(vertices, axis=0)

        # Create a mesh for each class
        for class_idx in unique_classes:
            # Get faces for this class
            class_mask = (faces_class == class_idx)
            if not np.any(class_mask):
                # Add None placeholder to maintain indexing
                self.gt_mesh_items.append(None)
                continue

            class_faces = faces[class_mask]

            # Create mesh data from triangles and faces
            md = gl.MeshData(vertexes=vertices, faces=class_faces)

            # Create mesh item with default properties
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=True,
                shader='shaded',
                drawFaces=True,
                drawEdges=False,
                glOptions='translucent'
            )

            # Store mesh item (don't add to GL widget yet)
            self.gt_mesh_items.append(mesh)

    def create_pre_meshes(self):
        """Create and cache Predicted mesh items"""
        if not hasattr(self, 'pre_mesh_items'):
            self.pre_mesh_items = []

        # Clear existing predicted meshes
        for item in self.pre_mesh_items:
            if item in self.gl_widget.items:
                self.gl_widget.removeItem(item)
        self.pre_mesh_items.clear()

        if self.image.get_pre_segmentation() is None:
            return

        # Get predicted mesh data
        vertices = self.mesh.get_verticesPre()
        faces = self.mesh.get_facesPre()
        faces_class = self.mesh.get_classesPre()

        if vertices is None or faces is None or faces_class is None:
            return

        # FIXED: Get actual unique classes from the segmentation, not from num_classes
        # num_classes might not include background (class 0)
        segmentation = self.image.get_pre_segmentation()
        actual_unique_classes = np.unique(segmentation)
        # Filter out background (0) and convert to list
        unique_classes = [int(c) for c in actual_unique_classes if c > 0]

        print(f"Debug: Creating meshes for classes: {unique_classes}")
        print(
            f"Debug: Number of unique classes in segmentation: {len(unique_classes)}")

        vertices = vertices - np.mean(vertices, axis=0)

        # Create a mesh for each class
        for class_idx in unique_classes:
            # Get faces for this class
            class_mask = (faces_class == class_idx)
            if not np.any(class_mask):
                # Add None placeholder to maintain indexing
                self.pre_mesh_items.append(None)
                continue

            class_faces = faces[class_mask]

            # Create mesh data from triangles and faces
            md = gl.MeshData(vertexes=vertices, faces=class_faces)

            # Create mesh item with default properties
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=True,
                shader='shaded',
                drawFaces=True,
                drawEdges=False,
                glOptions='translucent'
            )

            # Store mesh item (don't add to GL widget yet)
            self.pre_mesh_items.append(mesh)

        print(f"Debug: Created {len(self.pre_mesh_items)} mesh items")

    def on_segmentation_changed(self):
        """Optimized segmentation switching with cached meshes"""
        # Hide all current meshes
        for item in self.mesh_items:
            if item and item in self.gl_widget.items:
                self.gl_widget.removeItem(item)

        # Also hide any GT meshes if they're shown
        if hasattr(self, 'gt_mesh_items'):
            for item in self.gt_mesh_items:
                if item and item in self.gl_widget.items:
                    self.gl_widget.removeItem(item)

        # Switch to appropriate cached meshes
        if self.gt_radio.isChecked():
            if not hasattr(self, 'gt_mesh_items') or not self.gt_mesh_items:
                self.create_gt_meshes()
            self.mesh_items = self.gt_mesh_items
            self.comparison_mode = False

        elif self.pre_radio.isChecked():
            if not hasattr(self, 'pre_mesh_items') or not self.pre_mesh_items:
                self.create_pre_meshes()
            self.mesh_items = self.pre_mesh_items
            self.comparison_mode = False

        elif self.compare_radio.isChecked():
            # Comparison mode: show both
            if not hasattr(self, 'gt_mesh_items') or not self.gt_mesh_items:
                self.create_gt_meshes()
            if not hasattr(self, 'pre_mesh_items') or not self.pre_mesh_items:
                self.create_pre_meshes()
            self.mesh_items = self.pre_mesh_items  # Primary display is predicted
            self.comparison_mode = True

        # Apply colors and show meshes
        self.color_mesh()

        # Update the current slice views without changing slider positions
        self.update_current_slices()

    def invalidate_cached_meshes(self):
        """Call this method when segmentation data changes to clear cached meshes"""
        # Clear GT meshes
        if hasattr(self, 'gt_mesh_items'):
            for item in self.gt_mesh_items:
                if item and item in self.gl_widget.items:
                    self.gl_widget.removeItem(item)
            self.gt_mesh_items.clear()

        # Clear predicted meshes
        if hasattr(self, 'pre_mesh_items'):
            for item in self.pre_mesh_items:
                if item and item in self.gl_widget.items:
                    self.gl_widget.removeItem(item)
            self.pre_mesh_items.clear()

    def color_mesh(self):
        """Updated color_mesh method to work with cached meshes and comparison mode"""
        # Clear existing mesh items from GL widget
        for item in self.mesh_items:
            if item and item in self.gl_widget.items:
                self.gl_widget.removeItem(item)

        # Clear GT meshes too if they exist
        if hasattr(self, 'gt_mesh_items'):
            for item in self.gt_mesh_items:
                if item and item in self.gl_widget.items:
                    self.gl_widget.removeItem(item)

        # Check if we're in comparison mode
        if hasattr(self, 'comparison_mode') and self.comparison_mode:
            # COMPARISON MODE: Show both GT (transparent) and Predicted (solid)

            # First, add GT meshes with high transparency
            if hasattr(self, 'gt_mesh_items'):
                for i in range(len(self.enabeled_classes)):
                    if self.enabeled_classes[i] and i < len(self.gt_mesh_items) and self.gt_mesh_items[i] is not None:
                        qcolor = self.mesh_colors[i]
                        # GT with very high transparency (0.2 alpha)
                        rgba = (qcolor.redF(), qcolor.greenF(),
                                qcolor.blueF(), 0.2)
                        self.gt_mesh_items[i].setColor(rgba)
                        self.gl_widget.addItem(self.gt_mesh_items[i])

            # Then, add predicted meshes with solid color
            if hasattr(self, 'pre_mesh_items'):
                for i in range(len(self.enabeled_classes)):
                    if self.enabeled_classes[i] and i < len(self.pre_mesh_items) and self.pre_mesh_items[i] is not None:
                        qcolor = self.mesh_colors[i]
                        # Predicted with medium transparency (0.7 alpha) - solid but still see-through
                        rgba = (qcolor.redF(), qcolor.greenF(),
                                qcolor.blueF(), 0.7)
                        self.pre_mesh_items[i].setColor(rgba)
                        self.gl_widget.addItem(self.pre_mesh_items[i])
        else:
            # NORMAL MODE: Show only selected segmentation
            for i in range(len(self.enabeled_classes)):
                if self.enabeled_classes[i] and i < len(self.mesh_items) and self.mesh_items[i] is not None:
                    # Convert color name to RGBA
                    qcolor = self.mesh_colors[i]
                    rgba = (qcolor.redF(), qcolor.greenF(),
                            qcolor.blueF(), 0.7)
                    self.mesh_items[i].setColor(rgba)
                    self.gl_widget.addItem(self.mesh_items[i])

    def on_device_changed(self):
        """Handle device switching when user changes CPU/GPU radio button."""
        # Only proceed if a model is already loaded
        if self.network.model is None:
            # No model loaded yet - this is fine, just return
            return

        # Prevent recursive signal triggering
        if hasattr(self, '_switching_device') and self._switching_device:
            return

        self._switching_device = True

        try:
            # Get the desired device
            use_gpu = self.gpu_radio.isChecked()

            # Check if switching is supported for this model type
            if self.network.model_type == 'nnunet':
                # nnU-Net supports dynamic device switching
                self._switch_nnunet_device(use_gpu)

            elif self.network.model_type == 'monai':
                # MONAI models also support device switching
                self._switch_monai_device(use_gpu)

            elif self.network.model_type in ['pytorch', 'keras']:
                # PyTorch and Keras models can be moved, but warn user about potential issues
                self._switch_generic_device(use_gpu)

            else:
                # Unknown model type
                self.show_warning(
                    "Device switching is not supported for this model type.\n"
                    "Please reload the model to change the device."
                )
                # Revert radio button to current device state
                if self.network.processing_type == "gpu":
                    self.gpu_radio.setChecked(True)
                else:
                    self.cpu_radio.setChecked(True)

        finally:
            self._switching_device = False

    def _switch_nnunet_device(self, use_gpu):
        """Switch nnU-Net model to different device."""
        if use_gpu:
            if not torch.cuda.is_available():
                self.show_warning(
                    "GPU is not available on this system.\n"
                    "Keeping model on CPU."
                )
                self.cpu_radio.setChecked(True)
                return

            print("\n=== Switching nnU-Net to GPU ===")
            try:
                self.network.set_processing_type("gpu")
                device = torch.device("cuda")

                # Switch the appropriate handler
                if self.network.nnunet_version == 2 and self.network.nnunet_handler:
                    self.network.nnunet_handler.switch_device(device)
                elif self.network.nnunet_version == 1 and self.network.nnunet_v1_handler:
                    self.network.nnunet_v1_handler.switch_device(device)

                print("✓ Successfully switched to GPU")
                QMessageBox.information(
                    self,
                    "Device Switched",
                    "Model successfully switched to GPU!"
                )
            except Exception as e:
                print(f"✗ Error switching to GPU: {e}")
                self.show_warning(f"Failed to switch to GPU:\n{e}")
                self.cpu_radio.setChecked(True)

        else:  # Switch to CPU
            print("\n=== Switching nnU-Net to CPU ===")
            try:
                self.network.set_processing_type("cpu")
                device = torch.device("cpu")

                # Switch the appropriate handler
                if self.network.nnunet_version == 2 and self.network.nnunet_handler:
                    self.network.nnunet_handler.switch_device(device)
                elif self.network.nnunet_version == 1 and self.network.nnunet_v1_handler:
                    self.network.nnunet_v1_handler.switch_device(device)

                print("✓ Successfully switched to CPU")
                QMessageBox.information(
                    self,
                    "Device Switched",
                    "Model successfully switched to CPU!"
                )
            except Exception as e:
                print(f"✗ Error switching to CPU: {e}")
                self.show_warning(f"Failed to switch to CPU:\n{e}")
                self.gpu_radio.setChecked(True)

    def _switch_monai_device(self, use_gpu):
        """Switch MONAI model to different device."""
        if use_gpu:
            if not torch.cuda.is_available():
                self.show_warning(
                    "GPU is not available on this system.\n"
                    "Keeping model on CPU."
                )
                self.cpu_radio.setChecked(True)
                return

            print("\n=== Switching MONAI model to GPU ===")
            try:
                self.network.set_processing_type("gpu")
                device = torch.device("cuda")

                # Move MONAI model to GPU
                if self.network.model:
                    self.network.model.to(device)
                    self.network.model.eval()

                print("✓ Successfully switched to GPU")
                QMessageBox.information(
                    self,
                    "Device Switched",
                    "Model successfully switched to GPU!"
                )
            except Exception as e:
                print(f"✗ Error switching to GPU: {e}")
                self.show_warning(f"Failed to switch to GPU:\n{e}")
                self.cpu_radio.setChecked(True)

        else:  # Switch to CPU
            print("\n=== Switching MONAI model to CPU ===")
            try:
                self.network.set_processing_type("cpu")
                device = torch.device("cpu")

                # Move MONAI model to CPU
                if self.network.model:
                    self.network.model.to(device)
                    self.network.model.eval()

                print("✓ Successfully switched to CPU")
                QMessageBox.information(
                    self,
                    "Device Switched",
                    "Model successfully switched to CPU!"
                )
            except Exception as e:
                print(f"✗ Error switching to CPU: {e}")
                self.show_warning(f"Failed to switch to CPU:\n{e}")
                self.gpu_radio.setChecked(True)

    def _switch_generic_device(self, use_gpu):
        """Switch generic PyTorch/Keras model to different device."""
        if use_gpu:
            if not torch.cuda.is_available():
                self.show_warning(
                    "GPU is not available on this system.\n"
                    "Keeping model on CPU."
                )
                self.cpu_radio.setChecked(True)
                return

            reply = QMessageBox.question(
                self,
                "Switch to GPU?",
                "Device switching for this model type may not work correctly.\n\n"
                "It's recommended to reload the model instead.\n\n"
                "Do you want to try switching anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                # Keep current device
                if self.network.processing_type == "gpu":
                    self.gpu_radio.setChecked(True)
                else:
                    self.cpu_radio.setChecked(True)
                return

            try:
                self.network.set_processing_type("gpu")
                device = torch.device("cuda")

                if self.network.model_type == "pytorch":
                    self.network.model.to(device)
                    self.network.model.eval()

                print("✓ Switched to GPU")
                QMessageBox.information(
                    self,
                    "Device Switched",
                    "Attempted to switch to GPU.\n"
                    "If you encounter errors during inference, please reload the model."
                )
            except Exception as e:
                print(f"✗ Error switching to GPU: {e}")
                self.show_warning(f"Failed to switch to GPU:\n{e}")
                self.cpu_radio.setChecked(True)

        else:  # Switch to CPU
            reply = QMessageBox.question(
                self,
                "Switch to CPU?",
                "Device switching for this model type may not work correctly.\n\n"
                "It's recommended to reload the model instead.\n\n"
                "Do you want to try switching anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                # Keep current device
                if self.network.processing_type == "gpu":
                    self.gpu_radio.setChecked(True)
                else:
                    self.cpu_radio.setChecked(True)
                return

            try:
                self.network.set_processing_type("cpu")
                device = torch.device("cpu")

                if self.network.model_type == "pytorch":
                    self.network.model.to(device)
                    self.network.model.eval()

                print("✓ Switched to CPU")
                QMessageBox.information(
                    self,
                    "Device Switched",
                    "Attempted to switch to CPU.\n"
                    "If you encounter errors during inference, please reload the model."
                )
            except Exception as e:
                print(f"✗ Error switching to CPU: {e}")
                self.show_warning(f"Failed to switch to CPU:\n{e}")
                self.gpu_radio.setChecked(True)
