from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                               QListWidget, QListWidgetItem, QPushButton, 
                               QLabel, QDialogButtonBox)
from PySide6.QtCore import Qt


class DicomSeriesSelectionDialog(QDialog):
    """
    Dialog for selecting a DICOM series from a DICOMDIR file.
    """
    
    def __init__(self, series_dict, parent=None):
        """
        Initialize the dialog.
        
        :param series_dict: Dictionary with series UIDs as keys and info dict as values
                           Each info dict should contain 'description', 'number', and 'images'
        :param parent: Parent widget
        """
        super().__init__(parent)
        self.series_dict = series_dict
        self.selected_uid = None
        
        self.setWindowTitle("Select DICOM Series")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self._setup_ui()
        self._populate_list()
        
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Title label
        title_label = QLabel("Available DICOM Series:")
        title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title_label)
        
        # Series list widget
        self.series_list = QListWidget()
        self.series_list.setAlternatingRowColors(True)
        self.series_list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.series_list)
        
        # Info label
        self.info_label = QLabel("Select a series and click OK")
        self.info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.info_label)
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Connect selection change to update info
        self.series_list.currentItemChanged.connect(self._on_selection_changed)
        
    def _populate_list(self):
        """Populate the list widget with series information."""
        for uid, info in self.series_dict.items():
            series_number = info.get('number', 'N/A')
            description = info.get('description', 'No Description')
            num_images = len(info.get('images', []))
            
            # Create display text
            text = f"Series {series_number}: {description} ({num_images} images)"
            
            # Create list item
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, uid)  # Store UID in item data
            
            self.series_list.addItem(item)
        
        # Select first item by default
        if self.series_list.count() > 0:
            self.series_list.setCurrentRow(0)
    
    def _on_selection_changed(self, current, previous):
        """Handle selection change to update info label."""
        if current:
            uid = current.data(Qt.UserRole)
            info = self.series_dict[uid]
            num_images = len(info.get('images', []))
            self.info_label.setText(f"Selected series contains {num_images} images")
        else:
            self.info_label.setText("Select a series and click OK")
    
    def _on_double_click(self, item):
        """Handle double-click to accept immediately."""
        self._on_accept()
    
    def _on_accept(self):
        """Handle OK button click."""
        current_item = self.series_list.currentItem()
        if current_item:
            self.selected_uid = current_item.data(Qt.UserRole)
            self.accept()
        else:
            self.info_label.setText("Please select a series")
            self.info_label.setStyleSheet("color: red; font-style: italic;")
    
    def get_selected_series_uid(self):
        """
        Get the UID of the selected series.
        
        :return: Selected series UID or None if dialog was cancelled
        """
        return self.selected_uid
    
    @staticmethod
    def select_series(series_dict, parent=None):
        """
        Static method to show dialog and return selected series UID.
        
        :param series_dict: Dictionary with series UIDs as keys and info dict as values
        :param parent: Parent widget
        :return: Selected series UID or None if cancelled
        """
        dialog = DicomSeriesSelectionDialog(series_dict, parent)
        result = dialog.exec()
        
        if result == QDialog.Accepted:
            return dialog.get_selected_series_uid()
        return None