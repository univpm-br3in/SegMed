from PySide6.QtWidgets import (QPushButton, QVBoxLayout, QDialog,
                               QCheckBox, QHBoxLayout, QSpacerItem,
                               QSizePolicy, QGroupBox, QLineEdit, QLabel,
                               QMessageBox, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Signal


class PreProcessingWindow(QDialog):
    # Signal to emit the preprocessing options dictionary
    preprocessing_selected = Signal(dict)

    def __init__(self, parent=None):
        """Dialog for selecting preprocessing options."""
        super().__init__(parent)
        self.setWindowTitle("Preprocessing Selection")
        self.setMinimumSize(400, 400)

        # Store checkboxes and input fields for easy access
        self.checkboxes = {}
        self.input_fields = {}

        # Define mutually exclusive groups (options with same number can't be selected together)
        # You can modify these numbers to define which options are mutually exclusive
        self.exclusivity_groups = {
            "Z-Normalization (Based on This Image)": 1,
            "Z-Normalization (You need to enter avr/std)": 1
        }

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout()

        # Create group box for preprocessing options
        options_group = QGroupBox("Preprocessing Options")
        options_layout = QVBoxLayout()

        # Define preprocessing options and their input requirements
        preprocessing_options = [
            ("Z-Normalization (Based on This Image)", None, None),
            ("Z-Normalization (You need to enter avr/std)",
             "z_norm_values", "Mean, Std (e.g., 0.5, 0.2)"),
            ("Min/Max Normalization (Enter range)",
             "minmax_range", "Min, Max (e.g., 0, 1)"),
            ("Gaussian Blur", "blur_sigma", "Sigma value (e.g., 1.0)"),
            ("Isotropic Resampling", None, None),
        ]

        # Create checkboxes and input fields for each option
        for option_data in preprocessing_options:
            option_name = option_data[0]
            input_key = option_data[1]
            placeholder_text = option_data[2]

            # Create horizontal layout for checkbox and input field
            option_layout = QHBoxLayout()

            # Create checkbox
            checkbox = QCheckBox(option_name)
            checkbox.stateChanged.connect(
                lambda state, name=option_name: self.on_checkbox_changed(name, state))
            self.checkboxes[option_name] = checkbox
            option_layout.addWidget(checkbox)

            # Create input field if needed
            if input_key:
                input_field = QLineEdit()
                input_field.setPlaceholderText(placeholder_text)
                input_field.setEnabled(False)  # Initially disabled
                input_field.setMaximumWidth(200)
                self.input_fields[input_key] = input_field
                option_layout.addWidget(input_field)

                # Store the relationship between checkbox and input field
                checkbox.input_field = input_field
            else:
                checkbox.input_field = None

            options_layout.addLayout(option_layout)

        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # Add information about mutually exclusive options
        if self.exclusivity_groups:
            info_label = QLabel(
                "Note: Some options are mutually exclusive and cannot be selected together.")
            info_label.setStyleSheet("color: #666; font-size: 10px;")
            info_label.setWordWrap(True)
            main_layout.addWidget(info_label)

        # Add spacer to push button to bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum,
                             QSizePolicy.Policy.Expanding)
        main_layout.addItem(spacer)

        # Create button layout
        button_layout = QHBoxLayout()

        # Add spacer to center the button
        button_spacer_left = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Minimum)
        button_layout.addItem(button_spacer_left)

        # Create small OK button
        ok_button = QPushButton("OK")
        ok_button.setMaximumSize(80, 30)  # Make button small
        ok_button.clicked.connect(self.handle_ok_clicked)
        button_layout.addWidget(ok_button)

        # Add spacer to center the button
        button_spacer_right = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                          QSizePolicy.Policy.Minimum)
        button_layout.addItem(button_spacer_right)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def on_checkbox_changed(self, option_name, state):
        """Handle checkbox state changes."""
        checkbox = self.checkboxes[option_name]

        # Enable/disable associated input field
        if hasattr(checkbox, 'input_field') and checkbox.input_field:
            checkbox.input_field.setEnabled(state == 2)  # 2 means checked
            if state != 2:  # If unchecked, clear the input field
                checkbox.input_field.clear()

        # Handle mutual exclusivity
        if state == 2 and option_name in self.exclusivity_groups:  # If being checked
            group_id = self.exclusivity_groups[option_name]

            # Uncheck other options in the same exclusivity group
            for other_option, other_group_id in self.exclusivity_groups.items():
                if other_option != option_name and other_group_id == group_id:
                    other_checkbox = self.checkboxes[other_option]
                    if other_checkbox.isChecked():
                        other_checkbox.setChecked(False)
                        # Also disable and clear the input field if it exists
                        if hasattr(other_checkbox, 'input_field') and other_checkbox.input_field:
                            other_checkbox.input_field.setEnabled(False)
                            other_checkbox.input_field.clear()

    def validate_inputs(self):
        """Validate that required input fields are filled when their checkboxes are checked."""
        for option_name, checkbox in self.checkboxes.items():
            if checkbox.isChecked() and hasattr(checkbox, 'input_field') and checkbox.input_field:
                input_text = checkbox.input_field.text().strip()
                if not input_text:
                    QMessageBox.warning(self, "Input Required",
                                        f"Please enter values for '{option_name}' or uncheck the option.")
                    return False
        return True

    def parse_input_values(self, input_text, option_name):
        """Parse input values and return them as appropriate data types."""
        try:
            # Remove whitespace and split by comma
            values = [v.strip() for v in input_text.split(',')]

            # Convert to numbers (try float first, then int)
            parsed_values = []
            for value in values:
                try:
                    # Try float first
                    parsed_values.append(float(value))
                except ValueError:
                    # If float fails, it might be an integer or invalid
                    try:
                        parsed_values.append(int(value))
                    except ValueError:
                        raise ValueError(f"Invalid number format in '{value}'")

            return parsed_values
        except Exception as e:
            QMessageBox.warning(self, "Invalid Input",
                                f"Invalid input format for '{option_name}': {str(e)}")
            return None

    def handle_ok_clicked(self):
        """Handle OK button click - emit signal with selected options."""
        # Validate inputs first
        if not self.validate_inputs():
            return

        # Create dictionary with option states and values
        selected_options = {}

        for option_name, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                option_data = {"enabled": True}

                # If this option has an input field, get its value
                if hasattr(checkbox, 'input_field') and checkbox.input_field:
                    input_text = checkbox.input_field.text().strip()
                    if input_text:
                        # Find the input key for this option
                        input_key = None
                        for key, field in self.input_fields.items():
                            if field == checkbox.input_field:
                                input_key = key
                                break

                        if input_key:
                            parsed_values = self.parse_input_values(
                                input_text, option_name)
                            if parsed_values is None:  # Parsing failed
                                return
                            option_data["values"] = parsed_values
                            option_data["input_key"] = input_key

                selected_options[option_name] = option_data
            else:
                selected_options[option_name] = {"enabled": False}

        # Emit signal with the dictionary
        self.preprocessing_selected.emit(selected_options)

        # Close the dialog
        self.accept()

    def get_selected_options(self):
        """Get currently selected options with their values."""
        selected_options = {}

        for option_name, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                option_data = {"enabled": True}

                if hasattr(checkbox, 'input_field') and checkbox.input_field:
                    input_text = checkbox.input_field.text().strip()
                    if input_text:
                        parsed_values = self.parse_input_values(
                            input_text, option_name)
                        if parsed_values:
                            option_data["values"] = parsed_values

                selected_options[option_name] = option_data
            else:
                selected_options[option_name] = {"enabled": False}

        return selected_options
