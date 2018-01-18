import struct
import wave

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.signal import welch, butter, lfilter

# This script displays spectral cues of two wav files


CHUNK_SIZE = 4096
RATE = 44100
FORMAT = pyaudio.paInt16


def get_welch_spectrum(data):
    f, psd = welch(data, RATE)
    return f, psd


def get_spectrum(data):
    T = 1.0 / RATE
    N = data.shape[0]
    Pxx = (1. / N) * np.fft.rfft(data)
    f = np.fft.rfftfreq(N, T)
    return np.array(f[1:].tolist()), np.array((np.absolute(Pxx[1:])).tolist())


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# open files
stream_l = wave.open('/home/oesst/Dropbox/PhD/binaural head/recordings/full_head/simple_pinna_scaled_both_ear/fox_female_48dba/40_degree_left.wav', 'rb')
stream_r = wave.open('/home/oesst/Dropbox/PhD/binaural head/recordings/full_head/simple_pinna_scaled_both_ear/fox_female_48dba/40_degree_right.wav', 'rb')

# get number of frames in each file
frames_l = stream_l.getnframes()
frames_r = stream_r.getnframes()

# read data from files
block = stream_l.readframes(frames_l)
count = len(block) / 2
data_l = np.array(struct.unpack('%dh' % (count), block))

block = stream_r.readframes(frames_r)
count = len(block) / 2
data_r = np.array(struct.unpack('%dh' % (count), block))

# get welch spectrum
f_l, psd_l = get_welch_spectrum(data_l)
f_r, psd_r = get_welch_spectrum(data_r)

# make log scale
psd_l = np.log10(psd_l) * 20
psd_r = np.log10(psd_r) * 20

# plot

fig = plt.figure(2)

ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(f_l, psd_l)
ax1.set_ylabel('SPL FREE FIELD (dB)')

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(f_r, psd_r)
ax2.set_ylabel('SPL INSIDE (dB)')

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(f_l, psd_l - psd_r)
ax3.set_ylabel('Relative SPL (dB)')
ax3.set_xlabel('Frequency')

cor = np.correlate(psd_l - psd_r, psd_l - psd_r, 'same')

# fig = plt.figure(3)
# plt.plot(f_l,cor/np.max(np.abs(cor)))


plt.show()
