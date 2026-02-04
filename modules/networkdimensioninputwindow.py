from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QGroupBox,
    QRadioButton, QButtonGroup, QMessageBox
)
from PySide6.QtCore import Qt, Signal


class NetworkDimensionInputWindow(QDialog):
    """Dialog for manually inputting network dimensions when auto-detection fails."""
    
    # Signal to emit the configuration: (dimensions_tuple, channel_format, processing_mode)
    dimensions_configured = Signal(tuple, str, str)
    
    def __init__(self, parent=None):
        """Initialize the manual network dimension input dialog."""
        super().__init__(parent)
        self.setWindowTitle("Manual Network Dimension Input")
        self.setMinimumSize(500, 400)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout()
        
        # Information label
        info_label = QLabel(
            "The system could not automatically detect the network's input dimensions.\n"
            "Please provide the required information manually:"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-weight: bold; color: #A00000; margin-bottom: 10px;")
        main_layout.addWidget(info_label)
        
        # Network type group
        network_type_group = QGroupBox("Network Type")
        network_type_layout = QVBoxLayout()
        
        self.network_2d_radio = QRadioButton("2D Network")
        self.network_3d_radio = QRadioButton("3D Network")
        self.network_3d_radio.setChecked(True)  # Default to 3D
        
        self.network_type_group_btn = QButtonGroup()
        self.network_type_group_btn.addButton(self.network_2d_radio)
        self.network_type_group_btn.addButton(self.network_3d_radio)
        
        network_type_layout.addWidget(self.network_2d_radio)
        network_type_layout.addWidget(self.network_3d_radio)
        network_type_group.setLayout(network_type_layout)
        
        # Connect signal to update dimension input fields
        self.network_2d_radio.toggled.connect(self.on_network_type_changed)
        
        main_layout.addWidget(network_type_group)
        
        # Dimension input group
        dimension_group = QGroupBox("Input Dimensions")
        dimension_layout = QVBoxLayout()
        
        # Instruction label
        dim_instruction = QLabel("Enter the network's expected input dimensions:")
        dim_instruction.setStyleSheet("font-size: 10pt;")
        dimension_layout.addWidget(dim_instruction)
        
        # Dimension input fields
        dim_input_layout = QHBoxLayout()
        
        # Width
        dim_input_layout.addWidget(QLabel("Width:"))
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("e.g., 512")
        self.width_input.setMaximumWidth(100)
        dim_input_layout.addWidget(self.width_input)
        
        # Height
        dim_input_layout.addWidget(QLabel("Height:"))
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("e.g., 512")
        self.height_input.setMaximumWidth(100)
        dim_input_layout.addWidget(self.height_input)
        
        # Depth (only for 3D)
        self.depth_label = QLabel("Depth:")
        dim_input_layout.addWidget(self.depth_label)
        self.depth_input = QLineEdit()
        self.depth_input.setPlaceholderText("e.g., 128")
        self.depth_input.setMaximumWidth(100)
        dim_input_layout.addWidget(self.depth_input)
        
        dim_input_layout.addStretch()
        dimension_layout.addLayout(dim_input_layout)
        
        # Example label
        self.example_label = QLabel("Example: (512, 512, 128) for a 3D network")
        self.example_label.setStyleSheet("color: gray; font-style: italic; font-size: 9pt;")
        dimension_layout.addWidget(self.example_label)
        
        dimension_group.setLayout(dimension_layout)
        main_layout.addWidget(dimension_group)
        
        # Channel format group
        channel_group = QGroupBox("Channel Format")
        channel_layout = QVBoxLayout()
        
        channel_info = QLabel("How does the network expect channel information?")
        channel_info.setStyleSheet("font-size: 10pt;")
        channel_layout.addWidget(channel_info)
        
        self.channel_first_radio = QRadioButton("Channel First (e.g., PyTorch: [C, H, W] or [C, D, H, W])")
        self.channel_last_radio = QRadioButton("Channel Last (e.g., TensorFlow: [H, W, C] or [D, H, W, C])")
        self.channel_none_radio = QRadioButton("No Channel Dimension (grayscale only)")
        self.channel_first_radio.setChecked(True)  # Default to channel first
        
        self.channel_group_btn = QButtonGroup()
        self.channel_group_btn.addButton(self.channel_first_radio)
        self.channel_group_btn.addButton(self.channel_last_radio)
        self.channel_group_btn.addButton(self.channel_none_radio)
        
        channel_layout.addWidget(self.channel_first_radio)
        channel_layout.addWidget(self.channel_last_radio)
        channel_layout.addWidget(self.channel_none_radio)
        
        channel_group.setLayout(channel_layout)
        main_layout.addWidget(channel_group)
        
        # Processing mode group
        processing_group = QGroupBox("Image Processing Mode")
        processing_layout = QVBoxLayout()
        
        processing_info = QLabel("How should the image be adjusted to fit the network?")
        processing_info.setStyleSheet("font-size: 10pt;")
        processing_layout.addWidget(processing_info)
        
        self.pad_radio = QRadioButton("Pad (add borders to match dimensions)")
        self.resize_radio = QRadioButton("Resize (scale image to match dimensions)")
        self.pad_radio.setChecked(True)  # Default to padding
        
        self.processing_group_btn = QButtonGroup()
        self.processing_group_btn.addButton(self.pad_radio)
        self.processing_group_btn.addButton(self.resize_radio)
        
        processing_layout.addWidget(self.pad_radio)
        processing_layout.addWidget(self.resize_radio)
        
        processing_group.setLayout(processing_layout)
        main_layout.addWidget(processing_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setMaximumWidth(100)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        ok_button = QPushButton("OK")
        ok_button.setMaximumWidth(100)
        ok_button.clicked.connect(self.handle_ok_clicked)
        button_layout.addWidget(ok_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # Initialize UI state
        self.on_network_type_changed()
    
    def on_network_type_changed(self):
        """Handle changes to network type (2D/3D) selection."""
        is_3d = self.network_3d_radio.isChecked()
        
        # Enable/disable depth input based on network type
        self.depth_label.setEnabled(is_3d)
        self.depth_input.setEnabled(is_3d)
        
        if not is_3d:
            self.depth_input.clear()
            self.depth_input.setPlaceholderText("N/A")
        else:
            self.depth_input.setPlaceholderText("e.g., 128")
        
        # Update example label
        if is_3d:
            self.example_label.setText("Example: (512, 512, 128) for a 3D network")
        else:
            self.example_label.setText("Example: (512, 512) for a 2D network")
    
    def validate_inputs(self):
        """Validate that all required inputs are provided and valid."""
        # Validate width
        try:
            width = int(self.width_input.text().strip())
            if width <= 0:
                raise ValueError("Width must be positive")
        except (ValueError, AttributeError):
            QMessageBox.warning(
                self, "Invalid Input",
                "Please enter a valid positive integer for Width."
            )
            return False
        
        # Validate height
        try:
            height = int(self.height_input.text().strip())
            if height <= 0:
                raise ValueError("Height must be positive")
        except (ValueError, AttributeError):
            QMessageBox.warning(
                self, "Invalid Input",
                "Please enter a valid positive integer for Height."
            )
            return False
        
        # Validate depth (only for 3D networks)
        if self.network_3d_radio.isChecked():
            try:
                depth = int(self.depth_input.text().strip())
                if depth <= 0:
                    raise ValueError("Depth must be positive")
            except (ValueError, AttributeError):
                QMessageBox.warning(
                    self, "Invalid Input",
                    "Please enter a valid positive integer for Depth (required for 3D networks)."
                )
                return False
        
        return True
    
    def get_dimensions_tuple(self):
        """Get the dimensions as a tuple based on network type."""
        width = int(self.width_input.text().strip())
        height = int(self.height_input.text().strip())
        
        if self.network_3d_radio.isChecked():
            depth = int(self.depth_input.text().strip())
            return (width, height, depth)
        else:
            return (width, height)
    
    def get_channel_format(self):
        """Get the selected channel format."""
        if self.channel_first_radio.isChecked():
            return "channel_first"
        elif self.channel_last_radio.isChecked():
            return "channel_last"
        else:
            return "none"
    
    def get_processing_mode(self):
        """Get the selected processing mode."""
        if self.pad_radio.isChecked():
            return "pad"
        else:
            return "resize"
    
    def handle_ok_clicked(self):
        """Handle OK button click - validate and emit signal."""
        if not self.validate_inputs():
            return
        
        # Get all the configuration
        dimensions = self.get_dimensions_tuple()
        channel_format = self.get_channel_format()
        processing_mode = self.get_processing_mode()
        
        # Confirm with user
        confirm_msg = (
            f"Please confirm the configuration:\n\n"
            f"Network Type: {'3D' if self.network_3d_radio.isChecked() else '2D'}\n"
            f"Dimensions: {dimensions}\n"
            f"Channel Format: {channel_format}\n"
            f"Processing Mode: {processing_mode}\n\n"
            f"Proceed with segmentation?"
        )
        
        reply = QMessageBox.question(
            self, "Confirm Configuration",
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Emit signal with configuration
            self.dimensions_configured.emit(dimensions, channel_format, processing_mode)
            self.accept()
