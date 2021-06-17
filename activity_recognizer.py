import numpy.fft
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from enum import Enum
from PyQt5 import uic
import numpy as np
from sklearn import svm
from sklearn.exceptions import NotFittedError
from scipy import signal
from DIPPID import SensorUDP, SensorSerial, SensorWiimote
from DIPPID_pyqtnode import BufferNode, DIPPIDNode
import sys


class GestureNodeState(Enum):
    TRAINING = 1
    PREDICTING = 2
    IDLE = 3


class GestureRecognitionNode(Node):
    FEATURES = "features"
    OUTPUT = "output"

    nodeName = "GestureRecognition"

    state = GestureNodeState.IDLE

    gesture_dict = {}
    FEATURE_DATA = 'feature_data'
    gesture_id = 0
    recording = False
    TRAIN_LABEL_HELP_MESSAGE = "Select a gesture in the list, then press and hold 'R' on your \nkeyboard while" \
                               " performing the gesture to record a sample.\nRelease 'R' once you are done with" \
                               " the gesture"

    BUTTON_LAYOUT_HEIGHT = 100

    def __init__(self, name):
        terminals = {
            self.FEATURES: dict(io='in'),
            self.OUTPUT: dict(io='out')
        }

        self._init_ui()
        self.clf = svm.SVC(kernel='linear')

        Node.__init__(self, name, terminals=terminals)

    def process(self, **kwargs):
        if self.state == GestureNodeState.TRAINING:
            self.handle_training(kwargs)

        if self.state == GestureNodeState.PREDICTING:
            self.predict(kwargs)

    def _init_ui(self):
        # self.ui = uic.loadUi("gesture_node_ui.ui")
        self.ui = QtGui.QWidget()
        self.ui.keyPressEvent = keyPressEvent
        self.ui.keyReleaseEvent = keyReleaseEvent
        self.main_layout = QtGui.QVBoxLayout()
        self.button_layout = QtGui.QGridLayout()
        self.train_layout = QtGui.QHBoxLayout()
        self.gesture_list = QtGui.QListWidget()
        self.train_help_label = QtGui.QLabel()

        self.gesture_list.keyPressEvent = keyPressEvent
        self.gesture_list.keyReleaseEvent = keyReleaseEvent

        self._init_buttons()
        self._init_radio_buttons()

        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.edit_button)
        self.button_layout.addWidget(self.delete_button)

        self.train_layout.addWidget(self.train_button)
        self.train_layout.addWidget(self.predict_button)
        self.train_layout.addWidget(self.idle_button)

        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.train_layout)
        self.main_layout.addWidget(self.train_help_label)
        self.main_layout.addWidget(self.gesture_list)

        self.ui.setLayout(self.main_layout)

    def _init_buttons(self):
        self.add_button = QtGui.QPushButton("Add Gesture")
        self.edit_button = QtGui.QPushButton("Edit Gesture")
        self.delete_button = QtGui.QPushButton("Delete Gesture")

        self.add_button.clicked.connect(self._on_add_button_clicked)
        self.delete_button.clicked.connect(self._on_delete_button_clicked)

    def _init_radio_buttons(self):
        self.train_button = QtGui.QRadioButton("Train")
        self.predict_button = QtGui.QRadioButton("Predict")
        self.idle_button = QtGui.QRadioButton("Idle")

        self.train_button.clicked.connect(lambda: self.on_radio_button_clicked(self.train_button))
        self.predict_button.clicked.connect(lambda: self.on_radio_button_clicked(self.predict_button))
        self.idle_button.clicked.connect(lambda: self.on_radio_button_clicked(self.idle_button))

        self.idle_button.setChecked(True)

    def _on_add_button_clicked(self):
        self.add_gesture_window = uic.loadUi("add_gesture.ui")
        self.add_gesture_window.show()

        self.gesture_input = self.add_gesture_window.gestureNameInput
        button_box = self.add_gesture_window.buttonBox
        button_box.accepted.connect(self.on_new_gesture_added)

    def _on_delete_button_clicked(self):
        self.gesture_dict.pop(self.gesture_list.currentItem().text(), None)
        row = self.gesture_list.currentRow()
        self.gesture_list.takeItem(row)

    def on_radio_button_clicked(self, button):
        if button is self.train_button:
            self.state = GestureNodeState.TRAINING
            self.train_help_label.setText(self.TRAIN_LABEL_HELP_MESSAGE)

        elif button is self.predict_button:
            self.state = GestureNodeState.PREDICTING
            self.train_help_label.setText("")

        else:
            self.state = GestureNodeState.IDLE
            self.train_help_label.setText("")

    def ctrlWidget(self):
        return self.ui

    def on_new_gesture_added(self):
        self.gesture_dict[self.gesture_input.text()] = {}
        self.gesture_dict[self.gesture_input.text()]['id'] = self.gesture_id
        self.gesture_dict[self.gesture_input.text()][self.FEATURE_DATA] = []
        self.gesture_id += 1

        self.gesture_list.addItem(self.gesture_input.text())

    def recalculate_ui_height(self):
        self.ui.setFixedHeight(self.BUTTON_LAYOUT_HEIGHT + len(self.gesture_dict.keys()) * 50)
        self.update()

    def handle_training(self, kwargs):
        current_item = self.gesture_list.currentItem()
        if current_item is None:
            return

        if not self.recording:
            return

        gesture_name = current_item.text()
        features = kwargs[self.FEATURES]
        self.gesture_dict[gesture_name][self.FEATURE_DATA].append(features)

        fit_samples = []
        fit_targets = []

        for key in self.gesture_dict:
            for feature in self.gesture_dict[key][self.FEATURE_DATA]:
                feature = feature.flatten()
                fit_samples.append(feature)
                fit_targets.append(self.gesture_dict[key]['id'])

        if not all(p == fit_targets[0] for p in fit_targets):
            self.clf.fit(fit_samples, fit_targets)

    def predict(self, kwargs):
        features = kwargs[self.FEATURES]
        features = features.flatten()

        if len(features) == 51:  # TODO find out where this arbitrary number comes from and insert something that makes sense
            try:
                prediction = self.clf.predict([features])
            except NotFittedError:
                return

            for key in self.gesture_dict:
                if self.gesture_dict[key]['id'] == prediction[0]:
                    print(key)

    def set_recording(self, is_recording):
        self.recording = is_recording

        if self.state != GestureNodeState.TRAINING:
            return

        if is_recording:
            self.train_help_label.setText("Recording...")
        else:
            self.train_help_label.setText(self.TRAIN_LABEL_HELP_MESSAGE)


fclib.registerNodeType(GestureRecognitionNode, [('Assignment 8',)])


class FeatureExtractionFilter(Node):
    INPUT_X = "in_x"
    INPUT_Y = "in_y"
    INPUT_Z = "in_z"
    OUTPUT = "spectrograms"

    nodeName = "FeatureExtractionFilter"

    def __init__(self, name):
        terminals = {
            self.INPUT_X: dict(io='in'),
            self.INPUT_Y: dict(io='in'),
            self.INPUT_Z: dict(io='in'),
            self.OUTPUT: dict(io='out')
        }

        Node.__init__(self, name, terminals=terminals)

    def process(self, **kargs):
        fft_x = numpy.fft.fft(kargs[self.INPUT_X])
        fft_y = numpy.fft.fft(kargs[self.INPUT_Y])
        fft_z = numpy.fft.fft(kargs[self.INPUT_Y])

        # return {'fft': np.array([fft_x, fft_y, fft_z])}
        if len(kargs[self.INPUT_X]) == BUFFER_NODE_SIZE:
            spectrogram_x = signal.spectrogram(kargs[self.INPUT_X], nperseg=BUFFER_NODE_SIZE)
            spectrogram_y = signal.spectrogram(kargs[self.INPUT_Y], nperseg=BUFFER_NODE_SIZE)
            spectrogram_z = signal.spectrogram(kargs[self.INPUT_Z], nperseg=BUFFER_NODE_SIZE)

            return {'spectrograms': np.array([spectrogram_x[2], spectrogram_y[2], spectrogram_z[2]])}


fclib.registerNodeType(FeatureExtractionFilter, [('Assignment 8',)])


def connect_nodes():
    # DIPPID Nodes to Buffer Nodes
    fc.connectTerminals(dippid_node['accelX'], buffer_node_x['dataIn'])
    fc.connectTerminals(dippid_node['accelY'], buffer_node_y['dataIn'])
    fc.connectTerminals(dippid_node['accelZ'], buffer_node_z['dataIn'])

    # Buffer Nodes to FeatureExtraction
    fc.connectTerminals(buffer_node_x['dataOut'], feature_extraction_node[FeatureExtractionFilter.INPUT_X])
    fc.connectTerminals(buffer_node_y['dataOut'], feature_extraction_node[FeatureExtractionFilter.INPUT_Y])
    fc.connectTerminals(buffer_node_z['dataOut'], feature_extraction_node[FeatureExtractionFilter.INPUT_Z])

    fc.connectTerminals(feature_extraction_node[FeatureExtractionFilter.OUTPUT],
                        gesture_node[GestureRecognitionNode.FEATURES])


def keyPressEvent(event):
    if event.key() == QtCore.Qt.Key_R:
        if not event.isAutoRepeat():
            gesture_node.set_recording(True)


def keyReleaseEvent(event):
    if event.key() == QtCore.Qt.Key_R:
        if not event.isAutoRepeat():
            gesture_node.set_recording(False)

if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('DIPPIDNode demo')
    win.setMinimumWidth(450)
    win.setMinimumHeight(500)
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    BUFFER_NODE_SIZE = 32

    # Create an empty flowchart with a single input and output
    fc = Flowchart(terminals={})
    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    dippid_node = fc.createNode('DIPPID', pos=(0, -50))
    buffer_node_x = fc.createNode('Buffer', pos=(100, -100))
    buffer_node_y = fc.createNode('Buffer', pos=(100, -50))
    buffer_node_z = fc.createNode('Buffer', pos=(100, 0))
    feature_extraction_node = fc.createNode(FeatureExtractionFilter.nodeName, pos=(200, 50))
    gesture_node = fc.createNode(GestureRecognitionNode.nodeName, pos=(200, -50))

    connect_nodes()

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(QtGui.QApplication.instance().exec_())
