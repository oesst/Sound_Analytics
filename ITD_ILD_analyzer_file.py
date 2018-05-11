import struct
import wave

import numpy as np
import pyaudio
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from scipy.signal import butter, lfilter
from scipy.signal import correlate

# LEFT_RECORDING = '/home/oesst/cloudStore_UU/code_for_duc/sinus_2500Hz_pure_tone_2000ms.wav'
# RIGHT_RECORDING = '/home/oesst/cloudStore_UU/code_for_duc/sinus_2500Hz_pure_tone_2000ms_delayed_lower.wav'

RIGHT_RECORDING = '/home/oesst/ownCloud/PhD/Sounds/sinus_500Hz_wn_bg.wav'
LEFT_RECORDING = '/home/oesst/ownCloud/PhD/Sounds/sinus_500Hz_different_wn_bg_quarter_phase_shifted.wav'

class RealTimeSpecAnalyzer(pg.GraphicsWindow):
    def __init__(self):
        super(RealTimeSpecAnalyzer, self).__init__(title="Live FFT")
        self.pa = pyaudio.PyAudio()

        # CONSTANTS
        self.MONO = True # is the sound coming from 2 separate recordings -> True or simultaneously (one sound card) -> False?
        self.RATE = 44100
        self.CHUNK_SIZE = 4096
        self.FORMAT = pyaudio.paInt16
        self.TIME = 2  # time period to display
        self.INTENS_THRES = -50  # intensity threshold for ILD & ITD in db
        self.counter = 0
        self.logScale = False  # display frequencies in log scale

        # data storage
        self.data_l = np.zeros(int(self.RATE * self.TIME))
        self.data_r = np.zeros(int(self.RATE * self.TIME))
        if self.MONO:
            self.frequencies_l = np.zeros(int(self.CHUNK_SIZE / 2))
            self.frequencies_r = np.zeros(int(self.CHUNK_SIZE / 2))
        else:
            self.frequencies_l = np.zeros(int(self.CHUNK_SIZE ))
            self.frequencies_r = np.zeros(int(self.CHUNK_SIZE ))
        self.itds = np.zeros(100)  # store only the recent 100 values
        self.ilds = np.zeros(100)  # store only the recent 100 values
        self.timeValues = np.linspace(0, self.TIME, self.TIME * self.RATE)

        # initialization
        left_recording = LEFT_RECORDING
        right_recording = RIGHT_RECORDING
        self.open_files(left_recording, right_recording)
        # self.initMicrophones()
        self.initUI()

        # Timer to update plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        interval_ms = 1000 * (self.CHUNK_SIZE / self.RATE)
        print('Updating rate : %.1f ms' % interval_ms)
        self.timer.start(interval_ms)

    def initUI(self):
        # Setup plots
        self.setWindowTitle('ITD/ILD Analyzer')
        self.resize(1600, 1000)

        # first plot, signals amplitude
        self.p1 = self.addPlot(colspan=2)
        self.p1.setLabel('bottom', 'Time', 's')
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setTitle('')
        self.p1.setRange(xRange=(0, self.TIME), yRange=(-5.0, 5.0))
        # plot 2 signals in the plot
        self.ts_1 = self.p1.plot(pen=(1, 2))
        self.ts_2 = self.p1.plot(pen=(2, 2))

        self.nextRow()

        # # frequency of signal 1
        # self.p2 = self.addPlot(colspan=2)
        # self.p2.setLabel('bottom', 'Frequency LEFT', 'Hz')
        # self.spec_left = self.p2.plot(pen=(50, 100, 200),
        #                               brush=(50, 100, 200),
        #                               fillLevel=-100)
        # if self.logScale:
        #     self.p2.setRange(xRange=(0, 15000),
        #                      yRange=(-60, 20))
        #     self.spec_left.setData(fillLevel=-100)
        #     self.p2.setLabel('left', 'PSD', 'dB / Hz')
        # else:
        #     self.p2.setRange(xRange=(0, 15000),
        #                      yRange=(0, 100))
        #     self.spec_left.setData(fillLevel=0)
        #     self.p2.setLabel('left', 'PSD', '1 / Hz')
        #
        # self.nextRow()
        #
        # # frequency of signal 2
        # self.p3 = self.addPlot(colspan=2)
        # self.p3.setLabel('bottom', 'Frequency RIGHT', 'Hz')
        # self.spec_right = self.p3.plot(pen=(50, 100, 200),
        #                                brush=(50, 100, 200),
        #                                fillLevel=-100)
        # if self.logScale:
        #     self.p3.setRange(xRange=(0, 15000),
        #                      yRange=(-60, 20))
        #     self.spec_right.setData(fillLevel=-100)
        #     self.p3.setLabel('left', 'PSD', 'dB / Hz')
        # else:
        #     self.p3.setRange(xRange=(0, 15000),
        #                      yRange=(0, 100))
        #     self.spec_right.setData(fillLevel=0)
        #     self.p3.setLabel('left', 'PSD', '1 / Hz')
        #
        # self.nextRow()

        # write ITD & ILD in a box
        self.viewBox = self.addViewBox(colspan=2)
        self.textITD = pg.TextItem(text='The ITD is 0.0 ms', anchor=(-1.0, 6.0), border='w', fill=(255, 255, 255), color='#000000')
        self.viewBox.addItem(self.textITD)
        self.textILD = pg.TextItem(text='The ILD is 0.0 ', anchor=(-1.0, 5.0), border='w', fill=(255, 255, 255), color='#000000')
        self.viewBox.addItem(self.textILD)

        self.nextRow()

        # plot ITD and ILD bins
        p4 = self.addPlot(row=4, col=0, rowspan=1, colspan=1)
        self.histo_itd = p4.plot(stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        p5 = self.addPlot(row=4, col=1, rowspan=1, colspan=1)
        self.histo_ild = p5.plot(stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

    def open_files(self, file_left='', file_right=''):
        # if file is given read from file instead microphones
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


    def get_spectrum(self, data):
        T = 1.0 / self.RATE
        N = data.shape[0]
        Pxx = (1. / N) * np.fft.rfft(data)
        f = np.fft.rfftfreq(N, T)

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

    def cross_correlation_using_fft(self, x, y):
        from numpy.fft import fft, ifft, fft2, ifft2, fftshift

        f1 = fft(x)
        f2 = fft(np.flipud(y))
        cc = np.real(ifft(f1 * f2))
        return fftshift(cc)

    def find_delay(self, a, b, max_delay=0):
        # very accurate but not so fast as gcc
        # from scipy.signal import correlate
        # corr = correlate(a, b)
        # corr = np.correlate(a,b,'full')
        corr = self.cross_correlation_using_fft(a, b)
        # check only lags that are in range -max_delay and max_delay
        # print(corr)
        if max_delay:
            middle = np.int(np.ceil(len(corr) / 2))
            new_corr = np.zeros(len(corr))
            new_corr[middle - max_delay:middle + max_delay] = corr[middle - max_delay:middle + max_delay]
            lag = np.argmax(np.abs(new_corr)) - np.floor(len(new_corr) / 2)
        else:
            lag = np.argmax(np.abs(corr)) - np.floor(len(corr) / 2)

        return lag

    def update(self):
        try:
            data_l, data_r = self.readData()

            self.data_l = np.roll(self.data_l, -self.CHUNK_SIZE)
            self.data_r = np.roll(self.data_r, -self.CHUNK_SIZE)
            if self.MONO:
                self.data_l[-self.CHUNK_SIZE*1:] = data_l
                self.data_r[-self.CHUNK_SIZE*1:] = data_r
            else:
                self.data_l[-self.CHUNK_SIZE * 2:] = data_l
                self.data_r[-self.CHUNK_SIZE * 2:] = data_r
        except ValueError:
            # print('End of file reached. Stopping ....')
            left_recording = LEFT_RECORDING
            right_recording = RIGHT_RECORDING

            self.open_files(left_recording, right_recording)
            return
            # wait for 20 s to have a look at the data
            # exit(0)

        # get frequency spectrum
        f_l, psd_l = self.get_spectrum(data_l)
        f_r, psd_r = self.get_spectrum(data_r)

        # store frequency spectrum data for later use
        self.frequencies_l += psd_l
        self.frequencies_r += psd_r

        # plot data
        self.ts_1.setData(x=self.timeValues, y=self.data_l / 2 ** 15 * 5)
        self.ts_2.setData(x=self.timeValues, y=self.data_r / 2 ** 15 * 5)
        # self.spec_left.setData(x=f_l, y=(20 * np.log10(psd_l) if self.logScale else psd_l))
        # self.spec_right.setData(x=f_l, y=(20 * np.log10(psd_r) if self.logScale else psd_r))

        # get amplitude values in dB (theoretically it is not dB, since we don't have anything to compare to)
        signal_l = data_l / 2 ** 15
        signal_r = data_r / 2 ** 15
        intensities_l = np.log10(np.abs(signal_l)) * 20.0
        intensities_r = np.log10(np.abs(signal_r)) * 20.0

        # if any intensity exceeds as threshold calculate ITD and ILD
        if any(intensities_l > self.INTENS_THRES) or any(intensities_r > self.INTENS_THRES):

            # if counter is bigger than 100 -> reset it to 0
            if (self.counter >= 100):
                self.counter = 0

            # calculate ILD, use only frequencies between 1500 and 10000 Hz (indicies 138 til 927)
            signal_ild_l = np.fft.irfft(psd_l[138:927])
            signal_ild_r = np.fft.irfft(psd_r[138:927])
            ILD = 10 * np.log10(np.sum(signal_ild_l ** 2) / np.sum(signal_ild_r ** 2))
            # store values in counter index -> only recent 100 values
            self.ilds[self.counter] = ILD

            # calculate ITD, use only frequencies between 100 and 1000 Hz (indicies 8 til 91)
            # data_l = self.butter_bandpass_filter(data_l, 50, 1000, self.RATE, order=2)
            # data_r = self.butter_bandpass_filter(data_r, 50, 1000, self.RATE, order=2)

            lag = self.find_delay(data_l,data_r,50)
            # [lag, gphat] = self.gcc(data_l,data_r,200)
            ITD = lag/44100

            # [delay, gphat] = self.gcc(self.data_l, self.data_r)
            # ITD = delay / 44100

            # store values in counter index -> only recent 100 values
            if np.abs(ITD) > 0.0:
                self.itds[self.counter] = (ITD * 1000)

            # update textbox
            self.textILD.setPlainText('The ILD is %1.5f ' % ILD)
            self.textITD.setPlainText('The ITD is %1.5f ms' % (ITD * 1000))
            print('The ITD is %f ms and the ILD is %f' % ((ITD * 1000), ILD))

            # plot ITD as histogram
            y, x = np.histogram(self.itds, bins=np.linspace(-5.0, 5.0, 200))
            self.histo_itd.setData(x=x, y=y)
            # plot ILD as histogram
            y, x = np.histogram(self.ilds, bins=np.linspace(-25, 25, 500))
            self.histo_ild.setData(x=x, y=y)

            # increase counter
            self.counter += 1

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
