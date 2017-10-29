import struct
import wave

import numpy as np
import pyaudio
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from scipy.signal import butter, lfilter, welch

# This script calculates and displays the difference of the power spectral density for the left and right microphone offline
# from that we might be able to extract the spectral cues


class RealTimeSpecAnalyzer(pg.GraphicsWindow):
    def __init__(self):
        super(RealTimeSpecAnalyzer, self).__init__(title="Live FFT")

        # CONSTANTS
        self.RATE = 44100
        self.CHUNK_SIZE = 4096
        self.FORMAT = pyaudio.paInt16
        self.TIME = 2  # time period to display
        self.INTENS_THRES = -50  # intensity threshold for ILD & ITD in db
        self.counter = 0
        self.logScale = True  # display frequencies in log scale
        self.y_lim_spec = 150
        self.y_lim_spec_diff = 40

        # data storage
        self.data_l = np.zeros(self.RATE * self.TIME)
        self.data_r = np.zeros(self.RATE * self.TIME)
        self.frequencies_l = np.zeros(int(self.CHUNK_SIZE / 2))
        self.frequencies_r = np.zeros(int(self.CHUNK_SIZE / 2))
        self.itds = np.zeros(100)  # store only the recent 100 values
        self.ilds = np.zeros(100)  # store only the recent 100 values
        self.timeValues = np.linspace(0, self.TIME, self.TIME * self.RATE)

        # initialization
        left_recording = 'recordings/full_head/regular_audio_big_ear_right_no_ear_left_1m_front/white_noise_58dBA_-40_degree_left.wav'
        right_recording = 'recordings/full_head/regular_audio_big_ear_right_no_ear_left_1m_front/white_noise_58dBA_-40_degree_right.wav'
        self.open_files(left_recording, right_recording)
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
                         yRange=(-self.y_lim_spec, self.y_lim_spec))
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
                         yRange=(-self.y_lim_spec, self.y_lim_spec))
        self.spec_right_w.setData(fillLevel=0)
        self.p6.setLabel('left', 'PSD', '1 / Hz')

        self.nextRow()

        # frequency of signal 2
        self.p7 = self.addPlot(colspan=2)
        self.p7.setLabel('bottom', 'Interaural Spectral Difference', 'Hz')
        self.spec_diff_w = self.p7.plot(pen=(50, 100, 200),
                                        brush=(50, 100, 200))

        self.p7.setRange(xRange=(0, 15000),
                         yRange=(-self.y_lim_spec_diff, self.y_lim_spec_diff))
        self.p7.setLabel('left', 'SPL', 'db')

    def open_files(self, file_left, file_right):
        print('Reading data from file ...')
        self.stream_l = wave.open(file_left, 'rb')
        self.stream_r = wave.open(file_right, 'rb')

    def readData(self):
        # read data of first device

        block = self.stream_l.readframes(self.CHUNK_SIZE)
        count = len(block) / 2
        format = '%dh' % (count)
        data_l = struct.unpack(format, block)

        block = self.stream_r.readframes(self.CHUNK_SIZE)
        count = len(block) / 2
        format = '%dh' % (count)
        data_r = struct.unpack(format, block)
        return np.array(data_l), np.array(data_r)



    def get_welch_spectrum(self, data):
        f, psd = welch(data, self.RATE)
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

            psd_l_w = np.log10(psd_l_w) * 20
            psd_r_w = np.log10(psd_r_w) * 20

            # plot data
            self.ts_1.setData(x=self.timeValues, y=self.data_l)
            self.ts_2.setData(x=self.timeValues, y=self.data_r)

            self.spec_left_w.setData(x=f_l_w, y=psd_l_w)
            self.spec_right_w.setData(x=f_r_w, y=psd_r_w)
            self.spec_diff_w.setData(x=f_l_w, y=(psd_l_w - psd_r_w))

        except ValueError as ioerr:
            exit(0)





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
