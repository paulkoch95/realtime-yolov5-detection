import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

class MainApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera App!")
        button = QPushButton("Press Me!")
        self.setCentralWidget(button)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainApp()
    window.show()

    app.exec_()