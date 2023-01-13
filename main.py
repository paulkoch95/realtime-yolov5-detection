import sys
import time

from PySide6.QtCore import QSize, Qt, Signal, QThread, Slot, QTimer, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence
import cv2
import numpy as np
import torch

import time


from queue import Queue

model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')
print(device)
#exit(1)
model.to(device)


class InferenceStream(QThread):

    def __init__(self, image_queue: Queue, bbox_queue: Queue):
        super().__init__()
        self.image_queue: Queue = image_queue
        self.bbox_queue: Queue = bbox_queue

    def run(self):

        while True:
            try:
                img = self.image_queue.get(False)
                results = model(img, size=640)
                if self.bbox_queue.empty():
                    self.bbox_queue.put(results)
                # time.sleep(1)
            except Exception as e:
                print("Queue Empty", e)


class ImageStream(QThread):
    change_sig = Signal(np.ndarray)

    def __init__(self, bbox_queue):
        super().__init__()
        self.image_queue: Queue = bbox_queue


    def run(self):

        cap = cv2.VideoCapture(0)#, cv2.CAP_DSHOW)

        # codec = 0x47504A4D  # MJPG
        # cap.set(cv2.CAP_PROP_FPS, 30.0)
        # cap.set(cv2.CAP_PROP_FOURCC, codec)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_sig.emit(cv_img)
                if self.image_queue.empty():
                    self.image_queue.put(cv_img)
                # print(cv_img.shape)
                # time.sleep(.05)


class MainApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera App!")
        self.display_width = 1920
        self.display_height = 1080

        # self.setFixedWidth(self.display_width)
        # self.setFixedHeight(self.display_height)

        # self.setWindowFlag(self.windowFlags()& ~Qt.WindowMaximizeButtonHint)
        # self.showMaximized()

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

        self.image_queue = Queue(maxsize=1)
        self.box_queue = Queue(maxsize=1)

        # create the video capture thread
        self.thread = ImageStream(self.image_queue)
        self.inference = InferenceStream(self.image_queue, self.box_queue)

        self.inference.start()

        # connect its signal to the update_image slot
        self.thread.change_sig.connect(self.refresh)
        # start the thread
        self.thread.start()

        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.fps_label.hide)




    @Slot(np.ndarray)
    def refresh(self, cv_img):
        _img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        self.detect(rgb_image)
        self.fps_label.setText(f'frame time: {time.time()-start_time}')

        # cv2.putText(rgb_image, 'Christmas', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def detect(self, image):
        try:
            results = self.box_queue.get(False)
        except Exception as e:
            # print("no new data")
            return

        boxes = results.xyxy[0].tolist()
        # print(results.pandas().xyxy[0])
        labels = results.pandas().xyxy[0].name.tolist()


        # print(results.name)
        # Draw bounding boxes on the frame
        if boxes:
            # print(int(max(b[4] for b in boxes) * 100))
            for box, label in zip(boxes, labels):
                x2 = int(box[2])
                x1 = int(box[0])
                y2 = int(box[3])
                y1 = int(box[1])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # return image

if __name__ == "__main__":
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #print(torch.cuda.is_available())
    app = QApplication(sys.argv)

    window = MainApp()
    window.showMaximized()

    app.exec_()