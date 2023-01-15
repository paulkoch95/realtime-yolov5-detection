# System Imports
import queue
import sys
import time
from queue import Queue
#Module Imports
from PySide6.QtCore import Signal, QThread, Slot, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence, QColor
import cv2
import numpy as np
import torch

# load pytorch model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
# device = torch.device('cuda')
# device = torch.device('cpu')
device = torch.device('mps')
print(device)
# load model to GPU
model.to(device)


class InferenceStream(QThread):
    """
    Class responsible for PyTorch Inference on pre-trained model.
    """
    def __init__(self, image_queue: Queue, bbox_queue: Queue):
        super().__init__()
        self.image_queue: Queue = image_queue
        self.bbox_queue: Queue = bbox_queue

    def run(self):
        """
        Run the Inference on the PyTorch Model. This is a threaded operation, data is coming in form the image_queue
        in the form of cv2 Images (numpy.ndarray). Results are sent back to the bbox_qeueu which contains the whole
        PyTorch Results objects. This allows the inference to run independent from the display stream.
        :return: None
        """
        while True:
            try:
                # get cv2 image from queue, if queue is empty, skip inference.
                img = self.image_queue.get(False)
                # run the model (640 = full res for yolo v5l)
                results = model(img, size=640)
                if self.bbox_queue.empty():
                    # if there is no older result in the bbox queue, put new results in
                    self.bbox_queue.put(results)
            except queue.Empty:
                # Empty queue generates an error, which does not need to be handled in this case. We just skip inference
                # in the case of a full queue.
                pass


class ImageStream(QThread):
    """
    Class responsible to capture, pre-process and distribute images.
    """
    # PySide Signal (event) to indicate a new image is ready, also sending the image to all processes listening to this
    # signal.
    change_sig = Signal(np.ndarray)
    # Indication that Capture Device ID was changed in the GUI
    video_capture_device = Signal(int)


    def __init__(self, bbox_queue):
        super().__init__()
        # Queue where recevied images will be sent and then processed by the inference stream
        self.image_queue: Queue = bbox_queue

        self.cap = cv2.VideoCapture(0)
        self.video_capture_device.connect(self.change_input)

    def run(self) -> None:
        """
        Method which runs the capturing of the Video Input and puts the resulting images into the Signal for every image.
        And in case the Inference Stream is not busy, also puts the current image to the Inference to be processed.
        """

        # codec = 0x47504A4D  # MJPG
        # cap.set(cv2.CAP_PROP_FPS, 30.0)
        # cap.set(cv2.CAP_PROP_FOURCC, codec)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        while True:
            ret, cv_img = self.cap.read()
            if ret:
                # send image to GUI for display (done for every single frame)
                self.change_sig.emit(cv_img)
                # if Inference Stream is not busy, the current image is also send to the Inference Stream Class to be
                # processed using the YOLO V5 model inference.
                if self.image_queue.empty():
                    self.image_queue.put(cv_img)
                # time.sleep(.1)
    @Slot(int)
    def change_input(self, device: int) -> None:
        """
        Change Input Device of current cv2.Video Caputre
        :param device: ID of the device to switch to
        """
        self.cap = cv2.VideoCapture(device)

class AppSettingsDialog(QWidget):
    frame_time_signal = Signal(float)
    def __init__(self, parent=None):
        super(AppSettingsDialog, self).__init__(parent)
        # Setup to left settings window.
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAutoFillBackground(True)

        self.layout = QVBoxLayout()

        self.frame_time_label = QLabel("frame time")
        self.available_devices_label = QLabel("available devices")
        self.current_devices_label = QLabel("device")
        self.device_drop_down = QComboBox()

        self.list_devices()

        self.layout.addWidget(self.frame_time_label)
        self.layout.addWidget(self.available_devices_label)
        self.layout.addWidget(self.current_devices_label)
        self.layout.addWidget(self.device_drop_down)
        self.setLayout(self.layout)
        self.setFixedSize(self.layout.sizeHint())
        self.setFixedWidth(300)
        self.frame_time_signal.connect(self.set_label)

        self.emit_slot = Slot()

        self.device_drop_down.currentTextChanged.connect(self.set_device_name)

    def list_devices(self) -> None:
        """
        Helper function to check which devices are available to OpenCV.
        """
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
                self.device_drop_down.addItem(str(index))
            cap.release()
            index += 1
        self.available_devices_label.setText(f"Devices available: {str(arr)}")

    @Slot(str)
    def set_device_name(self, id: str) -> None:
        """
        GUI related stuff to display Information about the devices.
        """
        self.current_devices_label.setText(f"Device selected: {str(id)}")
        self.emit_slot.emit(int(id))

    @Slot(float)
    def set_label(self, frame_time):
        """
        Function to update label to show the current elapsed frame time in the GUI.
        :param frame_time: value to be display in fraction of seconds.
        """
        self.frame_time_label.setText(f"Frame Time: {frame_time * 1000:.3f}ms")
        self.frame_time_label.adjustSize()



class MainApp(QMainWindow):

    def __init__(self):
        super().__init__()
        # Setup Main Window if the App.
        self.setWindowTitle("Inference Results App")
        self.display_width = 1920
        self.display_height = 1080
        self.setMaximumWidth(self.display_width)
        self.setMaximumHeight(self.display_height)

        # Setup an empty label which will hold the QPixMap to display the Video Stream
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        # Additional GUI Widgets
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        self.frameless_toggle = QPushButton(self,"Frameless full screen")
        self.frameless_toggle.clicked.connect(self.toggle_framless)
        self.frameless_toggle.move(0,0)

        # Create two queues for threadsafe communication of images to the inference and results from the inference to
        # the drawing method.
        self.image_queue = Queue(maxsize=1)
        self.box_queue = Queue(maxsize=1)

        # Create the Threads for receiving images and running the inference
        self.thread = ImageStream(self.image_queue)
        self.inference = InferenceStream(self.image_queue, self.box_queue)

        # Create Settings Dialogue (top - left)
        self.settings = AppSettingsDialog(self)
        self.settings.emit_slot = self.thread.video_capture_device

        self.inference.start()

        # if a new image was recevied, display it to the label using the refresh Method.
        self.thread.change_sig.connect(self.refresh)
        self.thread.start()

        # Shortcut to hide and show the settings menue.
        QShortcut(QKeySequence(Qt.Key.Key_O), self, activated=self.toggle_settings)

        # Empty results list for the model inference results (will only be updated when inference has been run, which
        # allows for the good performance.
        self.results = []

    def toggle_framless(self):
        if self.windowState() & Qt.WindowFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()

    def toggle_settings(self):
        if self.settings.isVisible():
            self.settings.hide()
        else:
            self.settings.setVisible(True)

    @Slot(np.ndarray)
    def refresh(self, cv_img):
        """
        Helper Code to update the label with the new image from the camera.
        :param cv_img: Camera iMage whic is recevied from the Signal from the Image Thread.
        """
        _img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(_img)

    def convert_cv_qt(self, cv_img: np.ndarray) -> QPixmap:
        """
        Convert Numpy ND Array to QPixmap and draw the inference results.
        :param cv_img:
        :return:
        """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # Function which calls the drawing on the rgb_image object.
        self.redraw_inference_results(rgb_image)
        # standard conversion stuff, can be altered depending on needs.
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def redraw_inference_results(self, image) -> None:
        """
        Function which recives updated result data from model inference and updates the BBox Drawing.
        :param image: current image
        :return:
        """
        start_time = time.time()

        # if the model inference is done a new result object is available in the queue (can only hold 1 result at a time)
        # skips drawing if no data is available (only happens in the first iteration, from there on older data is always
        # available)
        if not self.box_queue.empty():
            self.results = self.box_queue.get(False)
        elif self.results == []:
            return

        # standard bounding box drawing
        boxes = self.results.xyxy[0].tolist()
        labels = self.results.pandas().xyxy[0].name.tolist()

        if boxes:
            for box, label in zip(boxes, labels):
                x2 = int(box[2])
                x1 = int(box[0])
                y2 = int(box[3])
                y1 = int(box[1])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # calculate how long the function to redraw took to measure drawing impact. usually in the 2-10ms range due to
        # all calls bei handed to the underlying C library of OpenCV
        t_diff = time.time() - start_time
        self.settings.frame_time_signal.emit(t_diff)

if __name__ == "__main__":
    # instantiate app
    app = QApplication(sys.argv)

    # launch maximized if possible
    window = MainApp()
    window.showMaximized()

    app.exec_()