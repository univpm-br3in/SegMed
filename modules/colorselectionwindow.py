from PySide6.QtWidgets import (
    QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QCheckBox, QScrollArea, QDialog, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor


class ColorSelectionWindow(QDialog):
    def __init__(self, parent=None, num_classes=15, initial_colors=None, initial_enabled=None, initial_names=None):
        """Dialog for selecting colors for each class.
        
        Args:
            parent: Parent widget
            num_classes: Number of classes to display
            initial_colors: List of initial color names (optional)
            initial_enabled: List of boolean values for enabled state (optional)
            initial_names: List of class names (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("Color Selection")
        self.setMinimumSize(600, 600)

        # Available colors (won't allow duplicates)
        self.available_colors = [
            "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
            "beige", "bisque", "black", "blanchedalmond", "blue",
            "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse",
            "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson",
            "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray",
            "darkgreen", "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen",
            "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen",
            "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet",
            "deeppink", "deepskyblue", "dimgray", "dimgrey", "dodgerblue",
            "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro",
            "ghostwhite", "gold", "goldenrod", "gray", "grey",
            "green", "greenyellow", "honeydew", "hotpink", "indianred",
            "indigo", "ivory", "khaki", "lavender", "lavenderblush",
            "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
            "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey", "lightpink",
            "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey",
            "lightsteelblue", "lightyellow", "lime", "limegreen", "linen",
            "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid",
            "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise",
            "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
            "navajowhite", "navy", "oldlace", "olive", "olivedrab",
            "orange", "orangered", "orchid", "palegoldenrod", "palegreen",
            "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru",
            "pink", "plum", "powderblue", "purple", "red",
            "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown",
            "seagreen", "seashell", "sienna", "silver", "skyblue",
            "slateblue", "slategray", "slategrey", "snow", "springgreen",
            "steelblue", "tan", "teal", "thistle", "tomato",
            "turquoise", "violet", "wheat", "white", "whitesmoke",
            "yellow", "yellowgreen"
        ]  # 148 colors

        self.used_colors = []
        self.num_classes = num_classes

        self.main_layout = QVBoxLayout()

        # Create a scroll area for many class selections
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        self.class_widgets = []

        # Create num_classes classes selection rows
        for i in range(num_classes):
            class_group = QWidget()
            class_layout = QHBoxLayout(class_group)

            # Enable checkbox
            enable_checkbox = QCheckBox()
            # Use initial_enabled if provided, otherwise default to True
            if initial_enabled is not None and i < len(initial_enabled):
                enable_checkbox.setChecked(initial_enabled[i])
            else:
                enable_checkbox.setChecked(True)

            # Color dropdown
            color_dropdown = QComboBox()
            color_dropdown.addItems(self.available_colors)
            
            # Use initial_colors if provided, otherwise use default
            if initial_colors is not None and i < len(initial_colors):
                color_name = initial_colors[i]
                default_index = self.available_colors.index(color_name) if color_name in self.available_colors else i
            else:
                default_index = min(i, len(self.available_colors) - 1)
            
            color_dropdown.setCurrentIndex(default_index)
            self.used_colors.append(self.available_colors[default_index])

            # Set text color preview in dropdown
            for j in range(color_dropdown.count()):
                color_name = color_dropdown.itemText(j)
                color_dropdown.setItemData(
                    j, QColor(color_name), Qt.BackgroundRole)

            # Connect signals
            enable_checkbox.toggled.connect(
                lambda checked, dd=color_dropdown: dd.setEnabled(checked))
            color_dropdown.currentIndexChanged.connect(
                lambda index, dd=color_dropdown, i=i: self.handle_color_change(
                    dd, i)
            )

            # Class name input
            name_input = QLineEdit()
            if initial_names is not None and i < len(initial_names):
                name_input.setText(initial_names[i])
            else:
                name_input.setText(f"Class_{i+1}")
            name_input.setPlaceholderText(f"Class {i+1} name")
            name_input.setMaximumWidth(150)

            # Label
            class_label = QLabel(f"Class {i+1}:")
            class_label.setMinimumWidth(60)

            # Add widgets to layout
            class_layout.addWidget(class_label)
            class_layout.addWidget(enable_checkbox)
            class_layout.addWidget(color_dropdown)
            class_layout.addWidget(name_input)

            self.class_widgets.append(
                (enable_checkbox, color_dropdown, class_label, name_input))
            scroll_layout.addWidget(class_group)

        # Add OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)

        self.main_layout.addWidget(scroll_area)
        self.main_layout.addWidget(ok_button)

        self.setLayout(self.main_layout)

    def handle_color_change(self, dropdown, class_index):
        new_color = dropdown.currentText()
        old_color = self.used_colors[class_index]

        # If the new color is already used by another class, swap them
        if new_color in self.used_colors and new_color != old_color:
            other_index = self.used_colors.index(new_color)
            if other_index != class_index:
                # Find the dropdown for the other class
                other_dropdown = self.class_widgets[other_index][1]

                # Find the index of the old color in the dropdown
                old_color_index = other_dropdown.findText(old_color)

                # Block signals to prevent recursion
                other_dropdown.blockSignals(True)
                other_dropdown.setCurrentIndex(old_color_index)
                other_dropdown.blockSignals(False)

                # Update used colors
                self.used_colors[other_index] = old_color

        # Update the used color for this class
        self.used_colors[class_index] = new_color

    def get_class_names(self):
        """Get the names for each class."""
        return [widget[3].text() for widget in self.class_widgets]
