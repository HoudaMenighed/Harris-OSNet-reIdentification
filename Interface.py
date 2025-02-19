from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QBrush, QFont
from PyQt5.QtCore import Qt
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Tracking and Re-ID Application")
        self.setGeometry(100, 100, 1200, 660)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # Title Widget
        self.title_widget = QWidget()
        self.title_layout = QHBoxLayout(self.title_widget)
        self.title_layout.setAlignment(Qt.AlignCenter)  # Center the text horizontally

        # Title Label
        self.title_label = QLabel("Multi-Object Tracking and Re-Identification", self)
        self.title_label.setFont(QFont("Arial", 45, QFont.Bold),)  # Set font size and style
        self.title_label.setAlignment(Qt.AlignCenter)  # Center align the text
        self.title_layout.addWidget(self.title_label)

        # Add Title Widget to Main Layout
        self.main_layout.addWidget(self.title_widget)

        # Create Buttons
        self.start_button = QPushButton("Start Tracking", self)
        self.stop_button = QPushButton("Stop Tracking", self)
        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.stop_button)

        # Example Label
        self.label = QLabel("Status: Waiting...", self)
        self.main_layout.addWidget(self.label)

        # Connect buttons to functions
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button.clicked.connect(self.stop_tracking)

        # Set Background Image
        self.background_image = "images/background2.jfif"
        self.update_background()

    def update_background(self):
        """Updates the background image to fit the window size."""
        o_image = QPixmap(self.background_image)
        s_image = o_image.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette = self.palette()
        brush = QBrush(s_image)
        palette.setBrush(self.backgroundRole(), brush)
        self.setPalette(palette)

    def resizeEvent(self, event):
        """Override the resize event to update the background image."""
        super().resizeEvent(event)
        self.update_background()

    def start_tracking(self):
        self.label.setText("Tracking started...")

    def stop_tracking(self):
        self.label.setText("Tracking stopped...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
