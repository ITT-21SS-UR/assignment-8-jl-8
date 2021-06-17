#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-

from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from DIPPID import SensorUDP, SensorSerial, SensorWiimote
import sys


class BufferNode(Node):
    """
    Buffers the last n samples provided on input and provides them as a list of
    length n on output.
    A spinbox widget allows for setting the size of the buffer.
    Default size is 32 samples.
    """
    nodeName = "Buffer"

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out'),
        }

        self.buffer_size = 32
        self._buffer = np.array([])
        Node.__init__(self, name, terminals=terminals)

    def process(self, **kwds):
        self._buffer = np.append(self._buffer, kwds['dataIn'])[-self.buffer_size:]

        return {'dataOut': self._buffer}

fclib.registerNodeType(BufferNode, [('Data',)])


class DIPPIDNode(Node):
    """
    Outputs sensor data from DIPPID supported hardware.

    Supported sensors: accelerometer (3 axis)
    Text input box allows for setting a Bluetooth MAC address or Port.
    Pressing the "connect" button tries connecting to the DIPPID device.
    Update rate can be changed via a spinbox widget. Setting it to "0"
    activates callbacks every time a new sensor value arrives (which is
    quite often -> performance hit)
    """

    nodeName = "DIPPID"

    def __init__(self, name):
        terminals = {
            'accelX': dict(io='out'),
            'accelY': dict(io='out'),
            'accelZ': dict(io='out'),
        }

        self.dippid = None
        self._acc_vals = []

        self._init_ui()

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_all_sensors)

        Node.__init__(self, name, terminals=terminals)

    def _init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()

        label = QtGui.QLabel("Port, BTADDR or TTY:")
        self.layout.addWidget(label)

        self.text = QtGui.QLineEdit()
        self.addr = "5700"
        self.text.setText(self.addr)
        self.layout.addWidget(self.text)

        label2 = QtGui.QLabel("Update rate (Hz)")
        self.layout.addWidget(label2)

        self.update_rate_input = QtGui.QSpinBox()
        self.update_rate_input.setMinimum(0)
        self.update_rate_input.setMaximum(60)
        self.update_rate_input.setValue(20)
        self.update_rate_input.valueChanged.connect(self.set_update_rate)
        self.layout.addWidget(self.update_rate_input)

        self.connect_button = QtGui.QPushButton("connect")
        self.connect_button.clicked.connect(self.connect_device)
        self.layout.addWidget(self.connect_button)
        self.ui.setLayout(self.layout)

    def update_all_sensors(self):
        if self.dippid is None or not self.dippid.has_capability('accelerometer'):
            return

        v = self.dippid.get_value('accelerometer')
        self._acc_vals = [v['x'], v['y'], v['z']]

        self.update()

    def update_accel(self, acc_vals):
        if not self.dippid.has_capability('accelerometer'):
            return

        self._acc_vals = [acc_vals['x'], acc_vals['y'], acc_vals['z']]
        self.update()

    def ctrlWidget(self):
        return self.ui

    def connect_device(self):
        if self.connect_button.text() != "connect" and self.connect_button.text() != "try again":
            return

        address = self.text.text().strip()
        self.connect_button.setText("connecting...")

        if '/dev/tty' in address: # serial tty
            self.dippid = SensorSerial(address)
        elif ':' in address:
            self.dippid = SensorWiimote(address)
        elif address.isnumeric():
            self.dippid = SensorUDP(int(address))
        else:
            print(f'invalid address: {address}')
            print('allowed types: UDP port, bluetooth address, path to /dev/tty*')

        if self.dippid is None:
            self.connect_button.setText("try again")
            return

        self.connect_button.setText("connected")
        self.set_update_rate(self.update_rate_input.value())
        self.connect_button.setEnabled(False)

    def set_update_rate(self, rate):
        if self.dippid is None:
            return

        self.dippid.unregister_callback('accelerometer', self.update_accel)

        if rate == 0:
            self.update_timer.stop()
        else:
            self.update_timer.start(int(1000 / rate))

    def process(self, **kwdargs):
        return {'accelX': np.array([self._acc_vals[0]]), 'accelY': np.array([self._acc_vals[1]]), 'accelZ': np.array([self._acc_vals[2]])}

fclib.registerNodeType(DIPPIDNode, [('Sensor',)])

if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('DIPPIDNode demo')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)

    # Create an empty flowchart with a single input and output
    fc = Flowchart(terminals={})
    w = fc.widget()

    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    pw1 = pg.PlotWidget()
    layout.addWidget(pw1, 0, 1)
    pw1.setYRange(0, 1)

    pw1Node = fc.createNode('PlotWidget', pos=(0, -150))
    pw1Node.setPlot(pw1)

    dippidNode = fc.createNode("DIPPID", pos=(0, 0))
    bufferNode = fc.createNode('Buffer', pos=(150, 0))

    fc.connectTerminals(dippidNode['accelX'], bufferNode['dataIn'])
    fc.connectTerminals(bufferNode['dataOut'], pw1Node['In'])

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(QtGui.QApplication.instance().exec_())
