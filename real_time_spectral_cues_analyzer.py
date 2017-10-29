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

        # return


        # CONSTANTS
        self.RATE = 44100
        self.CHUNK_SIZE = 4096
        self.FORMAT = pyaudio.paInt16
        self.TIME = 2  # time period to display
        self.logScale = False  # display frequencies in log scale
        self.y_lim_spec = 100
        self.y_lim_spec_diff = 35

        # data storage
        self.data_l = np.zeros(self.RATE * self.TIME)
        self.data_r = np.zeros(self.RATE * self.TIME)
        self.frequencies_l = np.zeros(int(self.CHUNK_SIZE / 2))
        self.frequencies_r = np.zeros(int(self.CHUNK_SIZE / 2))
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
        self.ts_2 = self.p1.plot(pen=(2, 2))

        self.nextRow()

        # frequency of signal 1
        self.p5 = self.addPlot(colspan=2)
        self.p5.setLabel('bottom', 'Frequency LEFT', 'Hz')
        self.spec_left_w = self.p5.plot(pen=(50, 100, 200),
                                        brush=(50, 100, 200),
                                        fillLevel=-100)

        self.p5.setRange(xRange=(0, 15000),
                         yRange=(0, self.y_lim_spec))
        self.spec_left_w.setData(fillLevel=0)
        self.p5.setLabel('left', 'PSD', '1 / Hz')

        self.nextRow()

        # frequency of signal 2
        self.p6 = self.addPlot(colspan=2)
        self.p6.setLabel('bottom', 'Frequency RIGHT', 'Hz')
        self.spec_right_w = self.p6.plot(pen=(50, 100, 200),
                                         brush=(50, 100, 200),
                                         fillLevel=-100)

        self.p6.setRange(xRange=(0, 15000),
                         yRange=(0, self.y_lim_spec))
        self.spec_right_w.setData(fillLevel=0)
        self.p6.setLabel('left', 'PSD', '1 / Hz')

        self.nextRow()

        # frequency of signal 2
        self.p7 = self.addPlot(colspan=2)
        self.p7.setLabel('bottom', 'Interaural Spectral Difference', 'Hz')
        self.spec_diff_w = self.p7.plot(pen=(50, 100, 200),
                                        brush=(50, 100, 200))

        self.p7.setRange(xRange=(0, 15000),
                         yRange=(-30,40))

    def initMicrophones(self):
        self.stream_l = self.pa.open(format=self.FORMAT,
                                     channels=1,
                                     rate=self.RATE,
                                     input=True,
                                     input_device_index=4,
                                     frames_per_buffer=self.CHUNK_SIZE)

        self.stream_r = self.pa.open(format=self.FORMAT,
                                     channels=1,
                                     rate=self.RATE,
                                     input=True,
                                     input_device_index=5,
                                     frames_per_buffer=self.CHUNK_SIZE)

    def readData(self):
        # read data of first device
        block = self.stream_l.read(self.CHUNK_SIZE)
        count = len(block) / 2
        format = '%dh' % (count)
        data_l = struct.unpack(format, block)

        # read data of first device
        block = self.stream_r.read(self.CHUNK_SIZE)
        count = len(block) / 2
        format = '%dh' % (count)
        data_r = struct.unpack(format, block)

        return np.array(data_l), np.array(data_r)

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
            data_l, data_r = self.readData()

            data_l = self.butter_bandpass_filter(data_l, 3000, 12000, self.RATE)
            data_r = self.butter_bandpass_filter(data_r, 3000, 12000, self.RATE)

            self.data_l = np.roll(self.data_l, -self.CHUNK_SIZE)
            self.data_l[-self.CHUNK_SIZE:] = data_l
            self.data_r = np.roll(self.data_r, -self.CHUNK_SIZE)
            self.data_r[-self.CHUNK_SIZE:] = data_r

            # get frequency spectrum
            # f_l, psd_l = self.get_spectrum(data_l)
            # f_r, psd_r = self.get_spectrum(data_r)
            # self.spec_left.setData(x=f_l, y=psd_l)
            # self.spec_right.setData(x=f_l, y=psd_r)
            # self.spec_diff.setData(x=f_l, y=(psd_r - psd_l))

            f_l_w, psd_l_w = self.get_welch_spectrum(data_l)
            f_r_w, psd_r_w = self.get_welch_spectrum(data_r)

            # change to log scale
            psd_l_w = np.log10(psd_l_w) * 20
            psd_r_w = np.log10(psd_r_w) * 20


            # plot data
            self.ts_1.setData(x=self.timeValues, y=self.data_l)
            self.ts_2.setData(x=self.timeValues, y=self.data_r)

            self.spec_left_w.setData(x=f_l_w, y=psd_l_w)
            self.spec_right_w.setData(x=f_r_w, y=psd_r_w)
            self.spec_diff_w.setData(x=f_l_w, y=(psd_r_w - psd_l_w))


        except IOError as ioerr:
            self.initMicrophones()
            # print(ioerr)
            pass

    def closeEvent(self, event):
        self.stream_l.close()
        self.stream_r.close()
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
