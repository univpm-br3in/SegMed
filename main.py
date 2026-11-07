import sys
from pathlib import Path

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen


def create_splash_screen():
    canvas = QPixmap(560, 320)
    canvas.fill(QColor("#F2F2F2"))

    painter = QPainter(canvas)
    logo_path = Path(__file__).resolve().parent / "resource" / "imgs" / "Segmed.png"
    logo = QPixmap(str(logo_path))

    if not logo.isNull():
        logo = logo.scaled(500, 235, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap((canvas.width() - logo.width()) // 2,
                           (270 - logo.height()) // 2, logo)
    else:
        painter.setPen(QColor("#A00000"))
        painter.setFont(QFont("Arial", 32, QFont.Bold))
        painter.drawText(QRect(0, 0, canvas.width(), 270), Qt.AlignCenter, "SegMed")

    painter.end()
    return QSplashScreen(canvas, Qt.WindowStaysOnTopHint)


def update_splash(splash, message):
    splash.showMessage(message, Qt.AlignBottom | Qt.AlignHCenter, QColor("#333333"))
    QApplication.processEvents()


def set_dark_theme(app):
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
    app = QApplication(sys.argv)
    set_dark_theme(app)

    splash = create_splash_screen()
    splash.show()
    update_splash(splash, "Loading SegMed modules...")

    try:
        import pyqtgraph as pg
        from modules.mainwindow import MainWindow

        pg.setConfigOptions(antialias=True)
        update_splash(splash, "Preparing the interface...")

        window = MainWindow()
        window.show()
        QApplication.processEvents()
        splash.finish(window)
    except Exception:
        splash.close()
        raise

    sys.exit(app.exec())
