import struct
import wave
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.signal import welch, butter, lfilter

# This script displays spectral cues extracted from several wav files


CHUNK_SIZE = 4096
RATE = 44100
FORMAT = pyaudio.paInt16
fig = plt.figure(0, figsize=(10, 20))
path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/simple_pinna_scaled_both_ear/'
num_of_neg_allevation = 1


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


# walk down each folder tree and display files for that

folders = [f for f in listdir()]

# get all wav files sorted
wav_files_only = [f for f in listdir(path) if isfile(join(path, f)) and (f[-3:] == 'wav')]
wav_files_only = sorted(wav_files_only)

# sorting is not correct for negative degrees, correct that manually
wav_files_only[0:2 * num_of_neg_allevation] = wav_files_only[0:2 * num_of_neg_allevation][::-1]
for i in range(0, 2 * num_of_neg_allevation, 2):
    wav_files_only[i], wav_files_only[i + 1] = wav_files_only[i + 1], wav_files_only[i]

first_notch_index = np.zeros(int(len(wav_files_only) / 2))
second_notch_index = np.zeros(int(len(wav_files_only) / 2))
psd_diff_all = []
for i in range(0, int(len(wav_files_only) / 2)):
    # skip every other loop -> only 20 degrees steps
    # if i%2 == 0:
    #     continue

    filename_l = wav_files_only[i * 2]
    filename_r = wav_files_only[i * 2 + 1]

    print("Opening files %s and %s" % (filename_r, filename_l))

    # open files
    stream_l = wave.open(path + filename_l, 'rb')
    stream_r = wave.open(path + filename_r, 'rb')

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

    # bandpass filter
    # data_l = butter_bandpass_filter(data_l,1000,12000,RATE)
    # data_r = butter_bandpass_filter(data_r,1000,12000,RATE)

    # # get spectrum
    # f_l,psd_l = get_spectrum(data_l)
    # f_r,psd_r = get_spectrum(data_r)
    #
    # # smooth
    # psd_l = savgol_filter(psd_l, 1001, 1)  # window size , polynomial order
    # psd_r = savgol_filter(psd_r, 1001, 1)  # window size , polynomial order

    # get welch spectrum
    f_l, psd_l = get_welch_spectrum(data_l)
    f_r, psd_r = get_welch_spectrum(data_r)

    # make log scale
    psd_l = np.log10(psd_l) * 20
    psd_r = np.log10(psd_r) * 20

    psd_diff = psd_l - psd_r

    # plot
    ax3 = fig.add_subplot(1, 1, 1)
    splitted = filename_l.split("_")
    ax3.plot(f_l, psd_diff + 50 * i, label=splitted[0] + ' ' + splitted[1])
    ax3.plot(f_l, np.ones(len(f_l)) * 30 * i, 'k--', )
    ax3.set_ylabel('Relative SPL (dB)')
    ax3.set_xlabel('Frequency (kHz)')

    # find first notch
    range_min_i = 15
    range_max_i = 40

    # if i > 4:
    #     range_min_i = 30
    #     range_max_i = 45
    # if i == 5:
    #     range_min_i = 30
    #     range_max_i = 35
    # if i == 6:
    #     range_min_i = 30
    #     range_max_i = 35

    x_min = np.argmin(psd_diff[range_min_i:range_max_i]) + range_min_i
    first_notch_index[i] = x_min
    ax3.arrow(f_l[x_min], psd_diff[x_min] + 50 * i - 15, 0, 10, head_width=220, head_length=5, fc='k', ec='k')

    psd_diff_all.append(psd_diff)

    # # find second notch
    # range_min_i = 50
    # range_max_i = 70
    # x_min = np.argmin(psd_diff[range_min_i:range_max_i]) + range_min_i
    # second_notch_index[i] = x_min

# change y tick  labels to 0
ax3.set_yticks(np.arange(int(len(wav_files_only) / 2)) * 30)
labels = ['0' for item in np.arange(int(len(wav_files_only) / 2))]
ax3.set_yticklabels(labels)
ax3.legend(loc='lower right', prop={'size': 20})

ax3.set_xticks(range(0, 18000, 1000))
ax3.set_xticklabels(range(0, 18))

fig = plt.figure(2)

ax = fig.add_subplot(4, 1, 1)
ax.plot(first_notch_index)
ax.set_title('First Notch Movement')
ax.set_ylabel('Notch Frequency Index')
ax.set_xticks(np.arange(int(len(wav_files_only) / 2)))
labels = [item.split('_')[0] + ' ' + item.split('_')[1] for item in wav_files_only[::2]]
ax.set_xticklabels(labels)
#
#
# ax = fig.add_subplot(4,1,2)
# ax.plot(second_notch_index)
# ax.set_title('Second Notch Movement')
# ax.set_ylabel('Notch Frequency Index')


fig = plt.figure(1)

i = 0
ii = 1
argmax = np.zeros(len(psd_diff_all) - 1)
for i in range(len(psd_diff_all) - 1):
    ii = i + 1

    result = np.correlate(psd_diff_all[i], -psd_diff_all[ii], mode='full')
    # result = result[int(result.size/2):]
    argmax[i] = -len(result) / 2 + np.argmax(result)

# plt.plot(argmax)

plt.plot(result / np.max(np.abs(result)))

fig = plt.figure(3)

plt.plot(psd_diff_all[i])
psd_diff_all[ii] = np.roll(psd_diff_all[ii], -int(len(result) / 2) - np.argmax(result))
plt.plot(psd_diff_all[ii])

plt.show()
