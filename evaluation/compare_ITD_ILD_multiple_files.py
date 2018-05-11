, ax2, ax3) import numpy as np
import wave
import struct
import pyaudio
from scipy.signal import welch, butter, lfilter, savgol_filter
import os as os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


CHUNK_SIZE = 4096
RATE = 44100
FORMAT = pyaudio.paInt16
path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_normalEars/'
# path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_simplePinna/'
# path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_rubberEars/'
# path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/clapping_hands_simplePinna/'
# path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/door-knock_simple_pinna/'
# path = '/home/oesst/cloudStore_UU/recordings_timo/recording_timo/whiteNoise_1_20000Hz_normalEars_fine/'
# path = '/home/oesst/cloudStore_UU/recordings_timo/whiteNoise_1_20000Hz_normalEars_JND/'
# path = '/home/oesst/cloudStore_UU/recordings_timo/recording_timo/whiteNoise_1_20000Hz_normalEars_JND_2/'
# path = '/home/oesst/cloudStore_UU/recordings_timo/recording_timo/whiteNoise_1_20000Hz_normalEars_JND_1deg_step/'
# path = '/home/oesst/cloudStore_UU/code_for_duc/recordings/sinus_500Hz_normalEars/'
# path = '/home/oesst/cloudStore_UU/code_for_duc/recordings/whiteNoise_1_20000Hz_normalEar_5steps/'
# path = '/home/oesst/cloudStore_UU/code_for_duc/recordings/sinus_2500hz_simplePinna/'
number_separators =3


# path = '/home/oesst/cloudStore_UU/recordings_noise_bursts_new_2/'

# here the index for the sound source location is defined (as in the CIPIC database).
# Starting at the most left (negative) azimuth to the most right (positive) azimuth.
# Starting at the lowest elevation (-negative) to the highest (positive) elevation.
# read_elevations = np.arange(0, 28)
# read_azimuths = np.arange(0, 19)

read_elevations = np.arange(0, 28)
read_azimuths = np.arange(0, 19)


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


def gcc(a, b,max_delay = 0):
    # Super fast but not so accurate as find_delay
    a_fft = np.fft.fft(a)
    b_fft = np.fft.fft(b)

    b_conj = b_fft.conj()

    nom = a_fft * b_conj

    denom = abs(nom)

    gphat = np.fft.ifft(nom / denom)

    delay = np.argmax(gphat)

    if max_delay:

        if delay > (len(a) / 2):
            delay = np.argmax(np.flip(gphat,0)[0:max_delay])
            delay =-delay
        else:
            delay = np.argmax(gphat[0:max_delay])

    return delay, gphat


def mesh_plot(fig, data, x_steps, y_steps):
    # data is a (n,3) array with all possible combinations of the data
    # x_steps & y_steps is the range of the x, y axis respectively. whereas n=x_steps*y_steps
    x = data[:, 1]
    y = data[:, 0]
    z = data[:, 2]

    x = np.linspace(min(x), max(x), x_steps)
    y = np.linspace(min(y), max(y), y_steps)
    x, y = np.meshgrid(x, y)
    z1 = np.reshape(z, [x_steps, y_steps]).T
    ax = fig.gca(projection='3d')
    # surf = ax.plot_wireframe(y, x, z1, rstride=1, cstride=1)
    surf = ax.plot_surface(x, y, z1, rstride=1, cstride=1, cmap=cm.jet, shade=False)
    # surf.set_facecolor((0,0,0,0))
    return ax


def cross_correlation_using_fft(x, y):
    from numpy.fft import fft, ifft, fft2, ifft2, fftshift

    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

def find_delay(a, b, max_delay=0):
    # very accurate but not so fast as gcc
    # from scipy.signal import correlate
    # corr = correlate(a, b)
    # corr = np.correlate(a,b,'full')
    corr = cross_correlation_using_fft(a,b)
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


###############################
#   Read all possible files   #
###############################

# read out azimuth directories
for root, dirs, files in os.walk(path, topdown=False):
    # sort dirs from neg to pos
    if dirs:
        dirs = np.array([f for f in dirs if 'azimuth' in f])
        azims = np.array([int(s.split('_')[1]) for s in dirs])
        indis = np.argsort(azims)
        dirs_sorted = dirs[indis]

# use only the azimuths specified in read_azimuths
azimuths = dirs_sorted[read_azimuths]
# from azimuths get all the elevations according to read_elevations, order them and store them
all_locations = np.empty((len(read_elevations), len(read_azimuths), 2), dtype='S100')
for i in range(0, len(azimuths)):
    d = azimuths[i]
    # order wav files by elevation
    wav_files_only = np.array([f for f in os.listdir(path + d) if os.path.isfile(os.path.join(path + d, f)) and (f[-3:] == 'wav')])
    elevs = np.array([int(s.split('_')[4 + number_separators]) for s in wav_files_only])
    indis = np.argsort(elevs)

    # wave files sorted but order (left,right) might be altered
    wav_files_only = wav_files_only[indis]

    for ii in range(0, len(wav_files_only), 2):
        if 'left' in wav_files_only[ii]:
            # swap that place with the next one
            wav_files_only[ii], wav_files_only[ii + 1] = wav_files_only[ii + 1], wav_files_only[ii]

    wav_files_only = np.array([azimuths[i] + '/' + file for file in wav_files_only])

    all_locations[:, i, 0] = wav_files_only[(read_elevations * 2)]
    all_locations[:, i, 1] = wav_files_only[(read_elevations * 2 + 1)]

wav_files_only = np.reshape(all_locations, [len(read_elevations) * len(read_azimuths), 2])

ITD_values = np.zeros((len(wav_files_only), 3))
ILD_values = np.zeros((len(wav_files_only), 3))
for i in range(0, int(wav_files_only.shape[0])):
    filename_l = wav_files_only[i, 0].decode('UTF-8')
    filename_r = wav_files_only[i, 1].decode('UTF-8')

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

    # get amplitude values in dB (theoretically it is not dB, since we don't have anything to compare to)
    signal_l = data_l / 2 ** 15
    signal_r = data_r / 2 ** 15

    # intensities_l = np.log10(np.abs(signal_l)) * 20.0
    # intensities_r = np.log10(np.abs(signal_r)) * 20.0

    signal_ild_l = signal_l
    signal_ild_r = signal_r

    ILD = 10 * np.log10(np.sum(signal_ild_l ** 2) / np.sum(signal_ild_r ** 2))

    # [delay, gphat] = gcc(data_l, data_r,250)
    # ITD = delay / RATE *1000
    delay = find_delay(data_l, data_r, 300)
    ITD = delay / RATE * 1000
    splitted = filename_l.split("_")
    azimuth = int(splitted[3 + number_separators])
    elevation = int(splitted[5 + number_separators])

    print('Azimuth : %s   Elevation : %s' % (azimuth, elevation))
    print('ITD : %f ' % ITD)
    print('ILD : %f ' % ILD)

    ITD_values[i, :] = [azimuth, elevation, ITD]
    ILD_values[i, :] = [azimuth, elevation, ILD]




plt.style.use('ggplot')


sound_type = filename_l.split("/")[1].split("_")[0:2]
# ### ITD vs. Azi vs. Ele ###
fig = plt.figure(figsize=(10,10))
ax = mesh_plot(fig, ITD_values, len(read_elevations), len(read_azimuths))
title = 'Normal Ears - White Noise'
ax.set_title(title)
ax.set_ylabel('Azimuth')
ax.set_yticks(ITD_values[0:len(read_azimuths):2,0])
ax.set_xlabel('Elevation')
ax.set_xticks(ITD_values[::len(read_azimuths)*2,1])
ax.set_zlabel('ITD (ms)')

ax.azim = 50
ax.elev = 30

fig = plt.figure(figsize=(10,10))
ax = mesh_plot(fig, np.flip(ILD_values,0), len(read_elevations), len(read_azimuths))
title = 'Normal Ears - White Noise'
ax.set_title(title)
ax.set_ylabel('Azimuth')
ax.set_yticks(ILD_values[0:len(read_azimuths):2,0])
ax.set_xlabel('Elevation')
ax.set_xticks(ILD_values[::len(read_azimuths)*2,1])
ax.set_zlabel('ILD (au)')

ax.azim = 50
ax.elev = 30


### ITD vs. Azi ###
fig = plt.figure()
# get all azimuth for 0 elevation
data = ITD_values[ITD_values[:, 1] == 0]
ax = plt.plot(data[:, 0], data[:, 2],linewidth=2.0)
plt.xlabel('Azimuth',fontweight='bold')
plt.ylabel('ITD (ms)',fontweight='bold')



# # ### ILD vs. Azi ###
# fig = plt.figure()
# # get all azimuth for 0 elevation
# data = ILD_values[ILD_values[:, 1] == 0]
# plt.plot(data[:, 0], data[:, 2])
#
plt.show()
