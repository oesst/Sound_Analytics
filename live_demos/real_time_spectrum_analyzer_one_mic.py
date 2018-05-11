import struct

import numpy as np
import pyaudio
import pyqtgraph as pg
import scipy.signal
from pyqtgraph.Qt import QtGui, QtCore
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


# This script calculates and displays the power spectral density for a single microphone online
# It displays the regular spectrum and the welch spectrum

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
        self.CHUNK_SIZE = 2048
        self.FORMAT = pyaudio.paInt16
        self.TIME = 5  # time period to display
        self.logScale = True  # display frequencies in log scale
        self.fft_bins = 128


        # data storage
        self.data_l = np.zeros(self.RATE * self.TIME)
        self.frequencies_l = np.zeros(int(self.CHUNK_SIZE / 2))
        self.timeValues = np.linspace(0, self.TIME, self.TIME * self.RATE)
        self.img_array = -np.ones((500, int(self.fft_bins / 2)))


        # initialization
        self.initMicrophones()
        self.initUI()

        # Timer to update plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        interval_ms = 1 * (self.CHUNK_SIZE / self.RATE)
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
                             yRange=(-60, 50))
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
                             yRange=(-60, 50))
            self.spec_left_welch.setData(fillLevel=-100)
            self.p2.setLabel('left', 'PSD', 'dB / Hz')
        else:
            self.p3.setRange(xRange=(0, 15000),
                             yRange=(0, 30))
            self.spec_left_welch.setData(fillLevel=0)
            self.p3.setLabel('left', 'PSD', '1 / Hz')

        self.nextRow()

        self.viewBox = self.addViewBox()
        ## lock the aspect ratio so pixels are always square
        self.viewBox.setAspectLocked(True)

        ## Create image item
        self.img = pg.ImageItem(border='w')
        self.viewBox.addItem(self.img)

        color = plt.cm.afmhot
        colors = color(range(0, 256))[:]
        pos = np.linspace(0,1,256)
        cmap = pg.ColorMap(pos, colors)
        lut = cmap.getLookupTable(0.0, 1.0, 256)


        self.img.setLookupTable(lut)
        self.img.setLevels([-1, 0])

        # freq = np.arange((self.CHUNK_SIZE / 2)) / (float(self.CHUNK_SIZE) / self.RATE)
        # freq = np.arange((self.CHUNK_SIZE / 2)) / (float(self.CHUNK_SIZE) / self.RATE)
        # yscale = 1.0 / (self.img_array.shape[1] / freq[-1])
        # self.img.scale((1. / self.RATE) * self.CHUNK_SIZE, yscale)

        self.win = np.hanning(self.CHUNK_SIZE)
        # self.show()

    def initMicrophones(self):
        self.stream_l = self.pa.open(format=self.FORMAT,
                                     channels=1,
                                     rate=self.RATE,
                                     input=True,
                                     frames_per_buffer=self.CHUNK_SIZE)

    def readData(self):
        # read data of first device
        block = self.stream_l.read(self.CHUNK_SIZE)
        count = len(block) / 2
        format = '%dh' % (count)
        data_l = struct.unpack(format, block)

        return np.array(data_l)

    def get_welch_spectrum(self, data,segs):
        f, psd = scipy.signal.welch(data, self.RATE,nperseg=segs)
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


    def overlap(self, X, window_size, window_step):
        """
        Create an overlapped version of X
        Parameters
        ----------
        X : ndarray, shape=(n_samples,)
            Input signal to window and overlap
        window_size : int
            Size of windows to take
        window_step : int
            Step size between windows
        Returns
        -------
        X_strided : shape=(n_windows, window_size)
            2D array of overlapped X
        """
        if window_size % 2 != 0:
            raise ValueError("Window size must be even!")
        # Make sure there are an even number of windows before stridetricks
        append = np.zeros((window_size - len(X) % window_size))
        X = np.hstack((X, append))

        ws = window_size
        ss = window_step
        a = X

        valid = len(a) - ws
        nw = (valid) // ss
        out = np.ndarray((int(nw), int(ws)), dtype=a.dtype)

        for i in np.arange(nw):
            # "slide" the window along the samples
            start = int(i * ss)
            stop = int(start + ws)
            out[int(i)] = a[start:stop]

        return out

    def stft(self,
             X,
             fftsize=128,
             step=65,
             mean_normalize=True,
             real=False,
             compute_onesided=True):
        """
        Compute STFT for 1D real valued input X
        """
        if real:
            local_fft = np.fft.rfft
            cut = -1
        else:
            local_fft = np.fft.fft
            cut = None
        if compute_onesided:
            cut = fftsize // 2
        if mean_normalize:
            X -= X.mean()

        X = self.overlap(X, fftsize, step)

        size = fftsize
        win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
        X = X * win[None]
        X = local_fft(X)[:, :cut]
        return X

    def pretty_spectrogram(self,
                           d,
                           log=True,
                           thresh=5,
                           fft_size=512,
                           step_size=64):
        """
        creates a spectrogram
        log: take the log of the spectrgram
        thresh: threshold minimum power for log spectrogram
        """
        specgram = np.abs(
            self.stft(
                d,
                fftsize=fft_size,
                step=step_size,
                real=False,
                compute_onesided=True))

        if log == True:
            specgram /= specgram.max()  # volume normalize to max 1
            specgram = np.log10(specgram)  # take log
            specgram[
                specgram <
                -thresh] = -thresh  # set anything less than the threshold as the threshold
        else:
            specgram[
                specgram <
                thresh] = thresh  # set anything less than the threshold as the threshold

        return specgram



    def update(self):
        try:
            data_l = self.readData()

            # data_l = self.butter_bandpass_filter(data_l, 0, 12000, self.RATE)

            self.data_l = np.roll(self.data_l, -self.CHUNK_SIZE)
            self.data_l[-self.CHUNK_SIZE:] = data_l

            # get frequency spectrum
            f_l, psd_l = self.get_spectrum(data_l)

            f_l_w, psd_l_w = self.get_welch_spectrum(data_l,127)

            # plot data
            self.ts_1.setData(x=self.timeValues, y=self.data_l)
            self.spec_left.setData(x=f_l, y=(20 * np.log10(psd_l)))
            self.spec_left_welch.setData(x=f_l_w, y=(20 * np.log10(psd_l_w)))

            psd = self.pretty_spectrogram(
                data_l.astype('float64'),
                log=True,
                thresh=1,
                fft_size=self.fft_bins,
                step_size=self.CHUNK_SIZE)


            # roll down one and replace leading edge with new data

            self.img_array = np.roll(self.img_array, -1, 0)
            self.img_array[-1:] = psd
            self.img.setImage(self.img_array, autoLevels=False)


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
