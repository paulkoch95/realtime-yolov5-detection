import sys
import time

from PySide6.QtCore import QSize, Qt, Signal, QThread, Slot, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


class ImageStream(QThread):
    change_sig = Signal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(1)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_sig.emit(cv_img)
                time.sleep(.15)


class MainApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera App!")
        self.display_width = 720
        self.display_height = 480

        self.setFixedWidth(self.display_width)
        self.setFixedHeight(self.display_height)

        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        self.fps_label = QLabel(self)
        self.fps_label.setText("Hallo Welt")
        self.fps_label.move(0, 0)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        # vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = ImageStream()
        # connect its signal to the update_image slot
        self.thread.change_sig.connect(self.refresh)
        # start the thread
        self.thread.start()

        self.timer = QTimer()
        self.timer.setInterval(100)  # 100 milliseconds = 0.1 seconds
        self.timer.timeout.connect(self.fps_display)  # Connect timeout signal to function
        self.timer.start()

    @Slot()  # Decorator to tell PyQt this method is a slot that accepts no arguments
    def fps_display(self):
        start_time = time.time()
        counter = 1
        # All the logic()
        time.sleep(0.1)
        time_now = time.time()
        fps = str((counter / (time_now - start_time)))
        self.fps_label.setText(fps)

    @Slot(np.ndarray)
    def refresh(self, cv_img):
        _img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = self.detect(rgb_image)
        # cv2.putText(rgb_image, 'Christmas', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def detect(self, image):
        results = model(image, size=320)
        boxes = results.xyxy[0].tolist()

        # Draw bounding boxes on the frame
        if boxes:
            # print(int(max(b[4] for b in boxes) * 100))
            for box in boxes:
                x2 = int(box[2])
                x1 = int(box[0])
                y2 = int(box[3])
                y1 = int(box[1])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.rectangle(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 2)

        return image

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainApp()
    window.show()

    app.exec_()