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


class GestureListItem(QtGui.QWidget):

    def __init__(self, parent=None):
        super(GestureListItem, self).__init__(parent)
        self.label = QtGui.QLabel("ee")
        self.button = QtGui.QPushButton()
        self.init_ui()
        self.show()

    def init_ui(self):
        item_layout = QtGui.QHBoxLayout()

        item_layout.addWidget(self.label)
        item_layout.addWidget(self.button)

        self.setLayout(item_layout)

    def set_label_text(self, text):
        self.label.setText(text)

    def set_button_text(self, text):
        self.button.setText(text)

    def set_button_icon(self, icon):
        self.button.setIcon(icon)

    def get_label_text(self):
        return self.label.text()


class GestureRecognitionNode(Node):
    """
    This node handles adding, training, editing, deleting and predicting gestures
    The user can add/edit/delete gestures to the list via the respective buttons
    Training data for gestures can then be recorded by swapping the node to "training" mode and performing the gesture
    while holding down the "R" key on a connected keyboard

    Training data for a single gesture can be reset by clicking the "Reset Gesture Training" button. This will only
    affect the gesture from that row.

    After adding and training at least two gestures, the "prediction" mode is functional. The matched gesture
    will be displayed on screen by :class:GestureTextWidget
    """
    FEATURES = "features"
    OUTPUT = "output"

    nodeName = "GestureRecognition"

    state = GestureNodeState.IDLE

    gesture_dict = {}
    output_val = {OUTPUT: "-"}
    FEATURE_DATA = 'feature_data'
    gesture_id = 0
    recording = False
    TRAIN_LABEL_HELP_MESSAGE = "Select a gesture in the list, then press and hold 'R' on your \nkeyboard while" \
                               " performing the gesture to record a sample.\nRelease 'R' once you are done with" \
                               " the gesture"
    NOT_ENOUGH_GESTURES_ADDED = "There are not enough gestures recorded.\nPlease add more gestures."
    NOT_ENOUGH_GESTURES_TRAINED = "There are not enough gestures with training data\nPlease train at least 2 gestures."
    SWITCH_TO_PREDICTION_MODE = "Switch to prediction mode to see the predicted performed gesture."

    BUTTON_LAYOUT_HEIGHT = 100

    def __init__(self, name):
        terminals = {
            self.FEATURES: dict(io='in'),
            self.OUTPUT: dict(io='out')
        }

        self._init_ui()
        self._init_confirm_windows()
        self.clf = svm.SVC(kernel='rbf')

        Node.__init__(self, name, terminals=terminals)

    def process(self, **kwargs):
        self.output_val = {self.OUTPUT: "-"}
        if self.state == GestureNodeState.TRAINING:
            self.handle_training(kwargs)

        if self.state == GestureNodeState.PREDICTING:
            self.output_val = self.predict(kwargs)

        self.set_special_output()

        return self.output_val

    def _init_confirm_windows(self):
        self.confirm_window_delete_gesture = uic.loadUi("confirm_action_ui.ui")
        self.confirm_window_delete_gesture.confirmButton.clicked.connect(self.delete_selected_gesture)
        self.confirm_window_delete_gesture.cancelButton.clicked.connect(lambda:
                                                                        self.confirm_window_delete_gesture.close())

        self.confirm_window_retrain_gesture = uic.loadUi("confirm_action_ui.ui")
        self.confirm_window_retrain_gesture.cancelButton.clicked.connect(
            lambda: self.confirm_window_retrain_gesture.close())

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
        self.edit_button.clicked.connect(self._on_edit_button_clicked)
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

    def _on_edit_button_clicked(self):
        self.edit_gesture_window = uic.loadUi("add_gesture.ui")
        self.edit_gesture_window.gestureNameInput.setText(
            self.gesture_dict[self.gesture_list.currentItem().identifier]['name'])

        self.edit_gesture_window.buttonBox.accepted.connect(self.on_gesture_edited)

        self.edit_gesture_window.show()

    def _on_delete_button_clicked(self):
        if self.confirm_window_delete_gesture is None:
            self.confirm_window_delete_gesture = uic.loadUi("confirm_action_ui.ui")

        if self.gesture_list.currentItem() is None:
            return

        self.confirm_window_delete_gesture.show()

    def on_gesture_edited(self):
        identifier = self.gesture_list.currentItem().identifier
        new_name = self.edit_gesture_window.gestureNameInput.text()
        self.gesture_list.itemWidget(self.gesture_list.currentItem()).set_label_text(new_name)

        self.gesture_dict[identifier]['name'] = new_name

    def delete_selected_gesture(self):
        self.gesture_dict.pop(self.gesture_list.currentItem().identifier, None)
        row = self.gesture_list.currentRow()
        self.gesture_list.takeItem(row)
        self.confirm_window_delete_gesture.close()

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
        self.gesture_dict[self.gesture_id] = {}
        self.gesture_dict[self.gesture_id]['name'] = self.gesture_input.text()
        self.gesture_dict[self.gesture_id][self.FEATURE_DATA] = []

        my_list_widget = GestureListItem()
        my_list_widget.set_label_text(self.gesture_input.text())
        my_list_widget.set_button_text(" Reset Gesture Training")

        # redo icon from flaticon.com - Free for personal and commercial purpose with attribution link:
        # https://www.flaticon.com/free-icon/redo_1828040?term=redo&page=1&position=2&page=1&position=2&related_id
        # =1828040&origin=tag
        my_list_widget.set_button_icon(QtGui.QIcon('redo.png'))

        list_item = QtGui.QListWidgetItem(self.gesture_list)
        list_item.setSizeHint(my_list_widget.sizeHint())
        list_item.identifier = self.gesture_id
        self.gesture_list.addItem(list_item)
        self.gesture_list.setItemWidget(list_item, my_list_widget)
        self.gesture_list.setCurrentItem(list_item)

        my_list_widget.button.clicked.connect(lambda: self.on_reset_gesture_training_clicked(list_item.identifier))
        self.gesture_id += 1

    def on_reset_gesture_training_clicked(self, gesture_id):
        self.confirm_window_retrain_gesture.confirmButton.clicked\
            .connect(lambda: self.reset_gesture_training(gesture_id))
        self.confirm_window_retrain_gesture.show()

    def reset_gesture_training(self, gesture_id):
        self.gesture_dict[gesture_id][self.FEATURE_DATA].clear()
        self.confirm_window_retrain_gesture.confirmButton.clicked.disconnect()
        self.confirm_window_retrain_gesture.close()

    def handle_training(self, kwargs):
        current_item = self.gesture_list.currentItem()
        if current_item is None:
            return

        if not self.recording:
            return

        features = kwargs[self.FEATURES]
        self.gesture_dict[current_item.identifier][self.FEATURE_DATA].append(features)

    def train(self):
        fit_samples = []
        fit_targets = []

        for key in self.gesture_dict:
            for feature in self.gesture_dict[key][self.FEATURE_DATA]:
                feature = feature.flatten()
                fit_samples.append(feature)
                fit_targets.append(key)

        if not all(p == fit_targets[0] for p in fit_targets):
            self.clf.fit(fit_samples, fit_targets)

    def predict(self, kwargs):
        features = kwargs[self.FEATURES]
        features = features.flatten()

        try:
            prediction = self.clf.predict([features])
        except NotFittedError:
            return

        for key in self.gesture_dict:
            if key == prediction[0]:
                return {self.OUTPUT: self.gesture_dict[key]['name']}

    def set_recording(self, is_recording):
        self.recording = is_recording

        if self.state != GestureNodeState.TRAINING:
            return

        if is_recording:
            self.train_help_label.setText("Recording...")
        else:
            self.train_help_label.setText(self.TRAIN_LABEL_HELP_MESSAGE)

    def set_special_output(self):
        if len(self.gesture_dict) >= 2:
            trained_gestures = 0
            for key in self.gesture_dict:
                if len(self.gesture_dict[key][self.FEATURE_DATA]) > 0:
                    trained_gestures += 1

            if trained_gestures >= 2:
                if self.state != GestureNodeState.PREDICTING:
                    self.output_val = {self.OUTPUT: self.SWITCH_TO_PREDICTION_MODE}
            else:
                self.output_val = {self.OUTPUT: self.NOT_ENOUGH_GESTURES_TRAINED}

        else:
            self.output_val = {self.OUTPUT: self.NOT_ENOUGH_GESTURES_ADDED}


fclib.registerNodeType(GestureRecognitionNode, [('Assignment 8',)])


class FeatureExtractionFilter(Node):
    """
    This node takes three inputs (accelerometer values via BufferNodes) and processes them via
    the signal.spectrogram function.
    The node output uses the third return value of the spectrogram function of each input.
    """
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

    fc.connectTerminals(gesture_node[GestureRecognitionNode.OUTPUT], text_node[DisplayTextNode.INPUT])


def keyPressEvent(event):
    if event.key() == QtCore.Qt.Key_R:
        if not event.isAutoRepeat():
            gesture_node.set_recording(True)


def keyReleaseEvent(event):
    if event.key() == QtCore.Qt.Key_R:
        if not event.isAutoRepeat():
            gesture_node.set_recording(False)
            gesture_node.train()


class GestureTextWidget(QtGui.QWidget):
    """
    Small widget for displaying information about the gesture on screen as a simple label
    """

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        self.layout = QtGui.QVBoxLayout()
        self.header_label = QtGui.QLabel("Currently predicted gesture: ")
        self.gesture_name_label = QtGui.QLabel("-")

        self.layout.addWidget(self.header_label)
        self.layout.addWidget(self.gesture_name_label)
        self.setLayout(self.layout)

        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.show()

    def set_text(self, text):
        self.gesture_name_label.setText(text)


class DisplayTextNode(Node):
    """
    Node that gets a single input and displays it on the widget set via set_widget
    In this application the widget is set to a GestureTextWidget, which displays the current gesture information
    on screen.
    """

    INPUT = 'in'
    widget = None

    nodeName = "DisplayTextNode"

    def __init__(self, name):
        terminals = {
            self.INPUT: dict(io='in'),
        }

        Node.__init__(self, name, terminals=terminals)

    def set_widget(self, widget):
        self.widget = widget

    def process(self, **kargs):
        self.widget.set_text(kargs[self.INPUT])


fclib.registerNodeType(DisplayTextNode, [('Assignment 8',)])

if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('DIPPIDNode demo')
    win.resize(800, 650)
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    BUFFER_NODE_SIZE = 32

    # Create an empty flowchart with a single input and output
    fc = Flowchart(terminals={})
    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    text_item = GestureTextWidget()
    text_node = fc.createNode(DisplayTextNode.nodeName, pos=(200, -100))
    text_node.set_widget(text_item)
    layout.addWidget(text_item, 0, 1)

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
