from scipy import signal, fft

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fftpack import irfft
from scipy.signal import welch,butter,lfilter,savgol_filter

RATE = 44100

def get_spectrum( data):
    T = 1.0 / RATE
    N = data.shape[0]
    Pxx = (1. / N) * np.fft.rfft(data)
    f = np.fft.rfftfreq(N, T)
    return np.array(f[1:].tolist()), np.array((np.absolute(Pxx[1:])).tolist())

# file_l = '/home/oesst/Desktop/subject21/0azleft.wav'
# file_r = '/home/oesst/Desktop/subject21/0azright.wav'
# data_l = sf.read(file_l)[0][8]
# data_r = sf.read(file_r)[0][8]

file_free = '/home/oesst/ownCloud/PhD/binaural head/recordings/recordings_free_field/azimuth_-90/free_field_azi_-90_ele_0_right.wav'

file_l = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/white_noise_bursts/azimuth_0/white_noise_bursts_azi_0_ele_0_left.wav'
# file_l = '/home/oesst/ownCloud/PhD/Code/Python/head_recording_control/recordings/whiteNoise_1_20000Hz_normalEars/azimuth_0/whiteNoise_1_20000Hz_normalEars_azi_0_ele_0_left.wav'
# file_l = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_simplePinna/azimuth_0/whiteNoise_1_20000Hz_azi_0_ele_0_left.wav'
data_l = sf.read(file_l)[0]
data_free = sf.read(file_free)[0]

data_l = data_l[20000:45000]
data_free = data_free[25000:50000]

hann = irfft(signal.hann(1000))
# filter the signals
filtered_signal = np.convolve(data_l,hann)
filtered_signal_free = np.convolve(data_free,hann)

freq_cut = 100

fig = plt.figure()
ax3 = fig.add_subplot(3, 1, 1)
ax3.plot(data_free)
ax3 = fig.add_subplot(3, 1, 2)
f_l,psd_l = get_spectrum(filtered_signal)
f_l,psd_free = get_spectrum(filtered_signal_free)
# remove freq above 17000Hz
f_l = f_l[0:freq_cut]
psd_l = psd_l[0:freq_cut]
psd_free = psd_free[0:freq_cut]
ax3.plot(f_l,psd_l/psd_free)
ax3 = fig.add_subplot(3, 1, 3)
f_l, psd_l = welch(filtered_signal,RATE)
f_l, psd_free = welch(filtered_signal_free,RATE)
f_l = f_l[0:freq_cut]
psd_l = psd_l[0:freq_cut]
psd_free = psd_free[0:freq_cut]
# psd_l = 10.**(psd_l/20.0)
ax3.plot(f_l,np.log(psd_l/psd_free))
# #
# #



fig = plt.figure()
ax3 = fig.add_subplot(3, 1, 1)
ax3.plot(data_l)
ax3 = fig.add_subplot(3, 1, 2)
f_l,psd_l = get_spectrum(data_l)
f_l,psd_free = get_spectrum(data_free)
f_l = f_l[0:freq_cut]
psd_l = psd_l[0:freq_cut]
psd_free = psd_free[0:freq_cut]
ax3.plot(f_l,psd_l/psd_free)
ax3 = fig.add_subplot(3, 1, 3)
f_l, psd_l = welch(data_l,RATE)
f_l, psd_free = welch(data_free,RATE)
f_l = f_l[0:freq_cut]
psd_l = psd_l[0:freq_cut]
psd_free = psd_free[0:freq_cut]
# psd_l = 10.**(psd_l/20.0)
ax3.plot(f_l,np.log(psd_l/psd_free))



plt.show()
