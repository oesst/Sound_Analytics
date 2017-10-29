import struct

import numpy as np
import pyaudio
import pyqtgraph as pg
import scipy.signal
from pyqtgraph.Qt import QtGui, QtCore
from scipy.signal import butter, lfilter


class RealTimeSpecAnalyzer(pg.GraphicsWindow):

    def __init__(self):
        super(RealTimeSpecAnalyzer, self).__init__(title="Live FFT")
        self.pa = pyaudio.PyAudio()

        p = self.pa
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            print((i, dev['name'], dev['maxInputChannels']))



        # CONSTANTS
        self.RATE = 44100
        self.CHUNK_SIZE = 4096
        self.FORMAT = pyaudio.paInt16
        self.TIME = 2  # time period to display
        self.logScale = True  # display frequencies in log scale

        # data storage
        self.data_l = np.zeros(self.RATE * self.TIME)
        self.frequencies_l = np.zeros(int(self.CHUNK_SIZE / 2))
        self.timeValues = np.linspace(0, self.TIME, self.TIME * self.RATE)

        # initialization
        self.initMicrophones()
        self.initUI()

        # Timer to update plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        interval_ms = 1000 * (self.CHUNK_SIZE / self.RATE)
        print('Updating graphs every %.1f ms' % interval_ms)
        self.timer.start(interval_ms)

    def initUI(self):
        # Setup plots
        self.setWindowTitle('Spectrum Analyzer')
        self.resize(1800, 800)

        # first plot, signals amplitude
        self.p1 = self.addPlot(colspan=2)
        self.p1.setLabel('bottom', 'Time', 's')
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setTitle('')
        self.p1.setRange(xRange=(0, self.TIME), yRange=(-8000, 8000))
        # plot 2 signals in the plot
        self.ts_1 = self.p1.plot(pen=(1, 2))

        self.nextRow()

        # frequency of signal 1
        self.p2 = self.addPlot(colspan=2)
        self.p2.setLabel('bottom', 'Frequency LEFT', 'Hz')
        self.spec_left = self.p2.plot(pen=(50, 100, 200))

        if self.logScale:
            self.p2.setRange(xRange=(0, 15000),
                             yRange=(-60, 20))
            self.spec_left.setData(fillLevel=-100)
            self.p2.setLabel('left', 'PSD', 'dB / Hz')
        else:
            self.p2.setRange(xRange=(0, 15000),
                             yRange=(0, 30))
            self.spec_left.setData(fillLevel=0)
            self.p2.setLabel('left', 'PSD', '1 / Hz')

        self.nextRow()

        # frequency of signal 1
        self.p3 = self.addPlot(colspan=2)
        self.p3.setLabel('bottom', 'Frequency LEFT', 'Hz')
        self.spec_left_welch = self.p3.plot(pen=(50, 100, 200))

        if self.logScale:
            self.p3.setRange(xRange=(0, 15000),
                             yRange=(-60, 20))
            self.spec_left_welch.setData(fillLevel=-100)
            self.p2.setLabel('left', 'PSD', 'dB / Hz')
        else:
            self.p3.setRange(xRange=(0, 15000),
                             yRange=(0, 30))
            self.spec_left_welch.setData(fillLevel=0)
            self.p3.setLabel('left', 'PSD', '1 / Hz')

    def initMicrophones(self):
        self.stream_l = self.pa.open(format=self.FORMAT,
                                     channels=1,
                                     rate=self.RATE,
                                     input=True,
                                     input_device_index=4,
                                     frames_per_buffer=self.CHUNK_SIZE)

    def readData(self):
        # read data of first device
        block = self.stream_l.read(self.CHUNK_SIZE)
        count = len(block) / 2
        format = '%dh' % (count)
        data_l = struct.unpack(format, block)

        return np.array(data_l)

    def get_welch_spectrum(self, data):
        f, psd = scipy.signal.welch(data, self.RATE)
        return f, psd

    def get_spectrum(self, data):
        T = 1.0 / self.RATE
        N = data.shape[0]
        Pxx = (1. / N) * np.fft.rfft(data)
        f = np.fft.rfftfreq(N, T)

        # remove first everything below 20Hz since microphones can't perceive that
        return np.array(f[1:].tolist()), np.array((np.absolute(Pxx[1:])).tolist())

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=6):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def update(self):
        try:
            data_l = self.readData()

            data_l = self.butter_bandpass_filter(data_l, 4000, 12000, self.RATE)

            self.data_l = np.roll(self.data_l, -self.CHUNK_SIZE)
            self.data_l[-self.CHUNK_SIZE:] = data_l

            # get frequency spectrum
            f_l, psd_l = self.get_spectrum(data_l)

            f_l_w, psd_l_w = self.get_welch_spectrum(data_l)

            # plot data
            self.ts_1.setData(x=self.timeValues, y=self.data_l)
            self.spec_left.setData(x=f_l, y=(20 * np.log10(psd_l)))
            self.spec_left_welch.setData(x=f_l_w, y=(20 * np.log10(psd_l_w)))


        except IOError as ioerr:
            self.initMicrophones()
            print(ioerr)
            pass

    def closeEvent(self, event):
        self.stream_l.close()
        self.pa.terminate()
        event.accept()  # let the window close


# QtGui.QApplication.setGraphicsSystem('opengl')
app = QtGui.QApplication([])

win = RealTimeSpecAnalyzer()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()