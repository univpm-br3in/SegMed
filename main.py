import sys
from PySide6.QtWidgets import QApplication
import pyqtgraph as pg
from modules.mainwindow import MainWindow

def set_dark_theme():
    app.setStyleSheet("""
        QWidget {
            background-color: #F2F2F2;
            color: black;
        }

        QPushButton {
            background-color: #A00000;
            color: white;
            border: 1px solid #777;
            border-radius: 5px;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #730d0d;
        }

        QCheckBox {
            spacing: 8px;
            color: black;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QCheckBox::indicator:unchecked {
            border: 1px solid #777;
            background-color: white;
        }
        QCheckBox::indicator:checked {
            border: 1px solid #777;
            background-color: #A00000;
            image: url(:/qt-project.org/styles/commonstyle/images/checkmark.png);
        }
        QRadioButton {
        spacing: 8px;
        color: black;
        }

        QRadioButton::indicator {
            width: 8px;
            height: 8px;
            border-radius: 8px;
        }

        QRadioButton::indicator:unchecked {
            border: 2px solid #777;
            background-color: white;
        }

        QRadioButton::indicator:unchecked:hover {
            border: 2px solid #A00000;
        }

        QRadioButton::indicator:checked {
            border: 2px solid #A00000;
            background-color: #A00000;
        }

        """)

if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    set_dark_theme()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
