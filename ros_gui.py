import imutils
import numpy as np
import rospy
import os
import std_msgs
import sys
import time
import argparse

from perception import DepthImage, BinaryImage, ColorImage
from gqcnn.msg import Observation, Action

from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import *

class RosGUI(QMainWindow):
    """A graphical user interface for Dex-Net using PyQt4 and ROS.
    """

    # Signals for callback interaction with main thread
    weight_signal = pyqtSignal(float)
    conf_signal = pyqtSignal(float)
    pph_signal = pyqtSignal(float)
    state_signal = pyqtSignal('QString')
    image_signal = pyqtSignal()
    success_signal = pyqtSignal(int)

    def __init__(self, screen):
        """Initialize the ROS GUI.
        """
        super(RosGUI, self).__init__()

        # Set constants
        self._total_time = 0
        self._n_attempts = 0
        self._n_picks = 0
        self._state = 'pause'
        self._img_dir = os.path.join(os.getcwd(), 'images')

        # Initialize the UI
        self._init_UI()
        self._current_image = None

        # Connect the signals to slots
        self.weight_signal.connect(self.set_weight)
        self.conf_signal.connect(self.set_confidence)
        self.pph_signal.connect(self.set_pph)
        self.state_signal.connect(self.set_state)
        self.image_signal.connect(self.update_central_image)
        self.success_signal.connect(self.set_success)

        # Load grasp images
        self._suction_image = ColorImage.open(self._image_fn('suction.png'))
        self._suction_image = self._suction_image.resize((40, 40))
        self._jaw_image = ColorImage.open(self._image_fn('gripper.png'))
        self._jaw_image = self._jaw_image.resize(0.06)
        self._push_image = ColorImage.open(self._image_fn('probe.png'))
        self._push_image = self._push_image.resize(0.2)
        self._push_end_image = ColorImage.open(self._image_fn('probe_end.png'))
        self._push_end_image = self._push_end_image.resize(0.06)

        # Initialize ROS subscribers
        self._init_subscribers()

        desktop = QApplication.desktop()
        if screen > desktop.screenCount():
            print 'Screen index is greater than number of available screens; using default screen'
        screenRect = desktop.screenGeometry(screen)
        self.move(screenRect.topLeft())
        self.setWindowFlags(Qt.X11BypassWindowManagerHint)
        self.setFixedSize(screenRect.size())
        self.show()

    def set_weight(self, w):
        """Set the weight field of the GUI.

        Parameters
        ----------
        w : float
            The weight of the bin in grams.
        """
        if w == 0:
            self._weight_bar.setMaximum(2000)
        elif not self._weight_initialized:
            self._weight_initialized = True
            self._weight_bar.setMaximum(w)
        self._weight_text.setText(str(int(w)))
        self._weight_bar.setValue(int(w))

    def set_confidence(self, c):
        """Set the confidence value of the GUI.

        Paramters
        ---------
        c : float
            The confidence of the current action, in [0,1].
        """
        self._conf_text.setText('{:.1f}'.format(100.0 * c))
        self._conf_bar.setValue(100.0 * c)

    def set_pph(self, p):
        """Set the picks-per-hour value of the GUI.

        Paramters
        ---------
        p : float
            The current PPH rate.
        """
        self._pph_text.setText(str(int(p)))

    def set_state(self, state):
        """Set the state of the GUI.

        Parameters
        ----------
        state : str
            One of 'pause', 'sensing', 'planning', 'moving', or 'error_recovery'.
        """
        if state == 'pause':
            self._pause_image.setPixmap(QPixmap(self._image_fn('pause_on.png')))
            self._state_image.setPixmap(QPixmap(self._image_fn('all_off.png')))
            self._error_image.setPixmap(QPixmap(self._image_fn('error_off.png')))
        elif state == 'sensing':
            self._pause_image.setPixmap(QPixmap(self._image_fn('pause_off.png')))
            self._state_image.setPixmap(QPixmap(self._image_fn('sensing.png')))
            self._error_image.setPixmap(QPixmap(self._image_fn('error_off.png')))
        elif state == 'planning':
            self._pause_image.setPixmap(QPixmap(self._image_fn('pause_off.png')))
            self._state_image.setPixmap(QPixmap(self._image_fn('computing.png')))
            self._error_image.setPixmap(QPixmap(self._image_fn('error_off.png')))
        elif state == 'moving':
            self._pause_image.setPixmap(QPixmap(self._image_fn('pause_off.png')))
            self._state_image.setPixmap(QPixmap(self._image_fn('moving.png')))
            self._error_image.setPixmap(QPixmap(self._image_fn('error_off.png')))
        elif state == 'error_recovery':
            self._pause_image.setPixmap(QPixmap(self._image_fn('pause_off.png')))
            self._state_image.setPixmap(QPixmap(self._image_fn('all_off.png')))
            self._error_image.setPixmap(QPixmap(self._image_fn('error_on.png')))

    def update_central_image(self):
        """Update the central image of the GUI.

        This updates the image from self._current_image.
        """
        img = self._current_image.data
        image = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        self._central_image.setPixmap(QPixmap(image))

    def set_success(self, success):
        """Set the success/failure state of the GUI.

        Parameters
        ----------
        success : int
            If 0, the most recent action was a failure.
            If 1, the most recent action was successful.
            If 2, the GUI should be cleared.
        """
        if success == 0:
            self._success_text.setText('FAILURE')
            self._success_text.setStyleSheet("border: 0px; color: white")
        elif success == 1:
            self._success_text.setText('SUCCESS')
            self._success_text.setStyleSheet("border: 0px; color: #000033")
        elif success == 2:
            self._success_text.setText('MOVING')
            self._success_text.setStyleSheet("border: 0px; color: #000033")

    def _init_subscribers(self):
        self._obs_sub = rospy.Subscriber('/bin_picking/observation', Observation, self._obs_callback)
        self._action_sub = rospy.Subscriber('/bin_picking/action', Action, self._action_callback)
        self._state_sub = rospy.Subscriber('/bin_picking/state', std_msgs.msg.String, self._state_callback)
        self._reward_sub = rospy.Subscriber('/bin_picking/reward', std_msgs.msg.Float32, self._reward_callback)
        self._weight_sub = rospy.Subscriber('/bin_picking/weight', std_msgs.msg.Float32, self._weight_callback)

    def _obs_callback(self, obs):
        depth_image = DepthImage(np.array(obs.image_data).reshape((obs.height, obs.width)))
        color_image = depth_image.inpaint(rescale_factor=0.25).to_color()
        self._current_image = color_image
        self.image_signal.emit()

    def _action_callback(self, action):
        # Update image with binary mask
        binary_data = np.frombuffer(action.mask_data, dtype=np.uint8).reshape(action.height, action.width)
        binary_image = BinaryImage(binary_data)
        mask = binary_image.nonzero_pixels()
        self._current_image.data[mask[:,0], mask[:,1], :] = (0.3 * self._current_image.data[mask[:,0], mask[:,1], :] +
                                                             0.7 * np.array([200, 30, 30], dtype=np.uint8))

        # Paint the appropriate action type over the current image
        action_type = action.action_type
        action_data = action.action_data

        if action_type == 'parallel_jaw':
            location = np.array([action_data[0], action_data[1]], dtype=np.uint32)
            axis = np.array([action_data[2], action_data[3]])
            if axis[0] != 0:
                angle = np.rad2deg(np.arctan((axis[1] / axis[0])))
            else:
                angle = 90
            jaw_image = ColorImage(imutils.rotate_bound(self._jaw_image.data, angle))
            self._current_image = self._overlay_image(self._current_image, jaw_image, location)
        elif action_type == 'suction':
            location = np.array([action_data[0], action_data[1]], dtype=np.uint32)
            self._current_image = self._overlay_image(self._current_image, self._suction_image, location)
        elif action_type == 'push':
            start = np.array([action_data[0], action_data[1]], dtype=np.int32)
            end = np.array([action_data[2], action_data[3]], dtype=np.int32)
            axis = (end - start).astype(np.float32)
            axis_len = np.linalg.norm(axis)
            if axis_len == 0:
                axis = np.array([0.0, 1.0])
                axis_len = 1.0
            axis = axis / axis_len

            if axis[0] != 0:
                angle = np.rad2deg(np.arctan2(axis[1], axis[0]))
            else:
                angle = (90.0 if axis[1] > 0 else -90.0)
            start_image = ColorImage(imutils.rotate_bound(self._push_image.data, angle))
            end_image = ColorImage(imutils.rotate_bound(self._push_end_image.data, angle))

            start = np.array(start - axis * 0.5 * self._push_image.width, dtype=np.int32)
            end = np.array(end + axis * 0.5 * self._push_end_image.width, dtype=np.int32)

            self._current_image = self._overlay_image(self._current_image, start_image, start)
            self._current_image = self._overlay_image(self._current_image, end_image, end)

        # Update display
        self.image_signal.emit()
        self.conf_signal.emit(action.confidence)

    def _state_callback(self, state):
        state = str(state.data)

        # Reset on pause
        if state == 'pause':
            self._total_time = 0
            self._n_attempts = 0
            self._n_picks = 0
            self._weight_initialized = False
            self.weight_signal.emit(0)
            self.success_signal.emit(True)
            self.conf_signal.emit(0.0)
        elif self._state == 'pause':
            self._start_time = time.time()

        if state == 'sensing':
            self.conf_signal.emit(0.0)
        if state == 'moving':
            self.success_signal.emit(2)

        self._state = state
        self.state_signal.emit(state)

    def _reward_callback(self, reward):
        self._n_attempts += 1

        # Set success
        success = reward.data > 0
        if success:
            self._n_picks += 1
        self.success_signal.emit(int(success))

        # Update PPH
        runtime = time.time() - self._start_time
        runtime = runtime / 3600.0
        self._total_time += runtime
        self._start_time = time.time()
        pph = self._n_picks / self._total_time
        self.pph_signal.emit(pph)

    def _weight_callback(self, weight):
        self.weight_signal.emit(weight.data)

    def _overlay_image(self, base, overlay, location):
        ow, oh = overlay.width, overlay.height
        bw, bh = base.width, base.height
        xmin, ymin = (location[0] - ow / 2), (location[1] - oh / 2)
        xmax, ymax = (location[0] + ow / 2 + ow % 2), (location[1] + oh / 2 + oh % 2)

        padding_l = (0 if (xmin > 0) else abs(xmin))
        padding_r = (0 if (xmax < bw) else (xmax - bw))
        padding_t = (0 if (ymin > 0) else abs(ymin))
        padding_b = (0 if (ymax < bh) else (ymax - bh))

        big_img = np.pad(base.data, ((padding_t, padding_b), (padding_l, padding_r), (0,0)), mode='constant')
        insert = big_img[ymin:ymax, xmin:xmax, :].copy()
        nzp = overlay.nonzero_pixels()
        insert[nzp[:,0], nzp[:,1], :] = overlay.data[nzp[:,0], nzp[:,1], :]
        big_img[ymin:ymax, xmin:xmax, :] = insert
        big_img = big_img[:bh, :bw, :]

        return ColorImage(big_img)

    def _image_fn(self, filename):
        x = os.path.join(self._img_dir, filename)
        return x

    def _init_UI(self):
        # Set font values
        font = QFont("Helvetica Neue", 95, QFont.Bold)
        small_font = QFont("Helvetica Neue", 33, QFont.Bold)
        success_font = QFont("Helvetica Neue", 55, QFont.Bold)

        # Weight Bar
        self._weight_initialized = False
        self._weight_bar = QProgressBar()
        self._weight_bar.setOrientation(QtCore.Qt.Vertical)
        style ="""
                QProgressBar {
                    border: 12px solid #000033;
                    border-radius: 12px;
                    width = 20px;
                }
                QProgressBar::chunk {
                    background-color: #ffcc00;
                }"""
        self._weight_bar.setStyleSheet(style)
        self._weight_bar.setMinimum(0)
        self._weight_bar.setMaximum(2000)
        self._weight_bar.setValue(0)
        self._weight_bar.setTextVisible(False)
        self._weight_bar.setFixedWidth(100)

        # Weight Box
        self._weight_text = QLabel("0")
        self._weight_text.setFont(font)
        self._weight_text.setStyleSheet("border: 1px solid #000033; background-color: #000033; color: white")
        self._weight_text.setAlignment(Qt.AlignCenter)

        weight_label = QLabel("GRAMS")
        weight_label.setStyleSheet("color: white")
        weight_label.setFont(small_font)
        weight_label.setAlignment(Qt.AlignCenter)

        weight_layout = QVBoxLayout()
        weight_layout.addWidget(self._weight_text)
        weight_layout.addWidget(weight_label)

        weight_widget = QWidget()
        weight_widget.setStyleSheet("border: 1px solid #000033; border-radius: 12px; background-color: #000033;")
        weight_widget.setLayout(weight_layout)


        # Confidence Bar
        self._conf_bar = QProgressBar()
        self._conf_bar.setOrientation(QtCore.Qt.Vertical)
        self._conf_bar.setStyleSheet(style)
        self._conf_bar.setMinimum(0)
        self._conf_bar.setMaximum(100)
        self._conf_bar.setValue(0)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setFixedWidth(100)

        # Confidence Box
        self._conf_text = QLabel("0.0")
        self._conf_text.setFont(font)
        self._conf_text.setStyleSheet("border: 1px solid #000033; background-color: #000033; color: white")
        self._conf_text.setAlignment(Qt.AlignCenter)

        conf_label = QLabel("% CONFIDENCE")
        conf_label.setStyleSheet("color: white")
        conf_label.setFont(small_font)
        conf_label.setAlignment(Qt.AlignCenter)

        conf_layout = QVBoxLayout()
        conf_layout.addWidget(self._conf_text, Qt.AlignCenter)
        conf_layout.addWidget(conf_label, Qt.AlignCenter)

        conf_widget = QWidget()
        conf_widget.setStyleSheet("border: 1px solid #000033; border-radius: 12px; background-color: #000033;")
        conf_widget.setLayout(conf_layout)

        # Picks Per Hour Box
        self._pph_text = QLabel("0")
        self._pph_text.setFont(font)
        self._pph_text.setStyleSheet("border: 1px solid #000033; background-color: #000033; color: white")
        self._pph_text.setAlignment(Qt.AlignCenter)

        pph_label = QLabel("PICKS PER HOUR")
        pph_label.setStyleSheet("color: white")
        pph_label.setFont(small_font)
        pph_label.setAlignment(Qt.AlignCenter)

        pph_layout = QVBoxLayout()
        pph_layout.addWidget(self._pph_text)
        pph_layout.addWidget(pph_label)

        pph_widget = QWidget()
        pph_widget.setStyleSheet("border: 1px solid #000033; border-radius: 12px; background-color: #000033;")
        pph_widget.setLayout(pph_layout)

        # Success/Failure Box
        self._success_text = QLabel("SUCCESS")
        self._success_text.setStyleSheet("border: 0px; color: #000033")
        #self._success_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._success_text.setAlignment(Qt.AlignCenter)
        self._success_text.setFont(success_font)

        success_layout = QVBoxLayout()
        success_layout.addWidget(self._success_text, Qt.AlignHCenter)

        success_widget = QWidget()
        success_widget.setStyleSheet("border: 0px solid #000033; background-color: #ffcc00; border-radius: 12px")
        success_widget.setLayout(success_layout)

        # header_widget
        header_widget = QLabel()
        header_widget.setPixmap(QPixmap(self._image_fn("header.png")))
        header_widget.setScaledContents(True)

        # Image Box
        self._central_image = QLabel()
        self._central_image.setPixmap(QPixmap(self._image_fn("camera.png")))
        self._central_image.setStyleSheet("border: 10px; color: #000033; border-radius: 12px")
        self._central_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._central_image.setAlignment(Qt.AlignCenter)

        image_layout = QVBoxLayout()
        image_layout.addWidget(self._central_image, Qt.AlignCenter)

        image_widget = QWidget()
        image_widget.setStyleSheet("border: 3px solid #000033; background-color: #000033; border-radius: 8px")
        image_widget.setLayout(image_layout)

        # Icons widget
        icons_widget = QLabel()
        icons_widget.setPixmap(QPixmap(self._image_fn("icons.png")))
        icons_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        icons_widget.setAlignment(Qt.AlignCenter)

        # State widgets
        self._state_image = QLabel()
        self._state_image.setPixmap(QPixmap(self._image_fn("all_off.png")))
        self._state_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._state_image.setAlignment(Qt.AlignCenter)

        self._pause_image = QLabel()
        self._pause_image.setPixmap(QPixmap(self._image_fn("pause_on.png")))
        self._pause_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._pause_image.setAlignment(Qt.AlignCenter)

        self._error_image = QLabel()
        self._error_image.setPixmap(QPixmap(self._image_fn("error_off.png")))
        self._error_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._error_image.setAlignment(Qt.AlignCenter)

        # Overall layout
        top_layout = QGridLayout()
        top_layout.setColumnStretch(0, 1)
        top_layout.setColumnStretch(1, 2)
        top_layout.setColumnStretch(2, 1)
        top_layout.setColumnMinimumWidth(0, 425)
        top_layout.setColumnMinimumWidth(2, 425)
        top_layout.setRowStretch(0, 1.25)
        top_layout.setRowStretch(1, 1.5)
        top_layout.setRowStretch(2, 7)
        top_layout.setRowStretch(3, 1)
        top_layout.setRowStretch(4, 1)
        top_layout.setRowStretch(5, 1)
        top_layout.addWidget(self._weight_bar, 2, 0, Qt.AlignHCenter)
        top_layout.addWidget(self._conf_bar, 2, 2, Qt.AlignHCenter)
        top_layout.addWidget(pph_widget, 0, 2, 2, 1)
        top_layout.addWidget(weight_widget, 3, 0, 2, 1)
        top_layout.addWidget(conf_widget, 3, 2, 2, 1)
        top_layout.addWidget(success_widget, 0, 0, 2, 1)
        top_layout.addWidget(header_widget, 0, 1, Qt.AlignCenter)
        top_layout.addWidget(image_widget, 1, 1, 3, 1, Qt.AlignCenter)
        top_layout.addWidget(icons_widget, 4, 1, Qt.AlignCenter)
        top_layout.addWidget(self._pause_image, 5, 0, Qt.AlignCenter)
        top_layout.addWidget(self._state_image, 5, 1, Qt.AlignCenter)
        top_layout.addWidget(self._error_image, 5, 2, Qt.AlignCenter)
        central_widget = QWidget()
        central_widget.setLayout(top_layout)
        self.setCentralWidget(central_widget)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GUI for Dex-Net Visualization')
    parser.add_argument('--screen', '-s', type=int, default=-1, help='screen number to display GUI on')
    args = parser.parse_args()

    # Initialize ROS node
    rospy.init_node('RosGUI', anonymous=True)

    # Initialize QT
    QApplication.setStyle("Motif")
    app = QApplication(sys.argv) 
    app.setStyleSheet('QMainWindow{background-color: white}')
    ex = RosGUI(args.screen)
    sys.exit(app.exec_())
