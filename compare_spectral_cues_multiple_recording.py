import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from matplotlib.font_manager import FontProperties
import pyaudio
from scipy.signal import welch,butter,lfilter,savgol_filter
from os import listdir
from os.path import isfile, join
from colour import Color

    # This script displays spectral cues extracted from several wav files



CHUNK_SIZE = 4096
RATE = 44100
FORMAT = pyaudio.paInt16
# path = '/home/oesst/cloudStore_UU/code_for_duc/recordings/door-knock_simple_pinna/azimuth_0/'
# path = '/home/oesst/cloudStore_UU/code_for_duc/recordings/clapping_hands_simplePinna/azimuth_0/'
path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_simplePinna/azimuth_0/'
path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_normalEars/azimuth_0/'
# path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_rubberEars/azimuth_0/'

dist_between_plots = 0.05
number_separators =1


def get_welch_spectrum(data):
    f, psd = welch(data, RATE,nperseg=256)
    return f, psd


def get_spectrum( data):
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



# ort wav files by elevation
wav_files_only = np.array([f for f in listdir(path) if isfile(join(path, f)) and (f[-3:] == 'wav') ])
elevs = np.array([ int(s.split('_')[6+number_separators]) for s in wav_files_only])
indis = np.argsort(elevs)

# wave files sorted but order (left,right) might be altered
wav_files_only = wav_files_only[indis]

for i in range(0,len(wav_files_only),2):
    if 'left' in wav_files_only[i]:
        # swap that place with the next one
        wav_files_only[i], wav_files_only[i+1] = wav_files_only[i+1], wav_files_only[i]




# create color gradient
red = Color("blue")
colors = list(red.range_to(Color("green"),int(len(wav_files_only)/2)))
first_notch_index = np.zeros(int(len(wav_files_only) / 2))
fig = plt.figure(figsize=(10,10))
for i in range(0,int(len(wav_files_only)/2)):
# for i in range(0, 1):

    filename_l = wav_files_only[i*2]
    filename_r = wav_files_only[i*2+1]

    print("Opening files %s and %s" % (filename_r,filename_l))

    data_l = sf.read(path+filename_r)[0]
    data_r = sf.read(path+filename_l)[0]
    data_l = sf.read('/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/recordings_free_field/azimuth_-90/free_field_azi_-90_ele_0_right.wav')[0]

    # use only one burst
    # data_l = data_l[20000:45000]
    # data_r = data_r[20000:45000]

    # # bandpass filter
    # data_l = butter_bandpass_filter(data_l,1000,12000,RATE)
    # data_r = butter_bandpass_filter(data_r,1000,12000,RATE)
    #
    #
    #
    # # # get spectrum
    # f_l,psd_l = get_spectrum(data_l)
    # f_r,psd_r = get_spectrum(data_r)
    # # #
    # # # smooth
    # psd_l = savgol_filter(psd_l, 1001, 2)  # window size , polynomial order
    # psd_r = savgol_filter(psd_r, 1001, 2)  # window size , polynomial order

    # # get welch spectrum
    f_l,psd_l = get_welch_spectrum(data_l)
    f_r,psd_r = get_welch_spectrum(data_r)

    # make log scale
    psd_l = np.log10(psd_l)*20
    psd_r = np.log10(psd_r)*20

    # cut off frequencies at the end (>18000Hz)
    indis = f_l < 15000

    f_l = f_l[indis]
    f_r = f_r[indis]
    psd_l = psd_l[indis]
    psd_r = psd_r[indis]


    # psd_diff = psd_l-psd_r
    psd_diff = psd_l/psd_r
    # psd_diff = np.log(psd_diff)


    # plot
    ax3 = fig.add_subplot(1,1,1)
    splitted = filename_l.split("_")
    ax3.plot(f_l,psd_diff +dist_between_plots*i, label=splitted[5+number_separators]+' '+splitted[6+number_separators],linewidth=3.0,color=colors[i].get_rgb())
    ax3.plot(f_l,np.ones(len(f_l))+dist_between_plots*i, 'k--',)
    ax3.set_ylabel('Relative SPL (dB)',fontweight='bold')
    ax3.set_xlabel('Frequency (kHz)',fontweight='bold')

    range_min_i = 20
    range_max_i = 40

    # # find first notch
    # range_min_i = 30
    # range_max_i = 70
    #
    # if i > 4:
    #     range_min_i = 60
    #     range_max_i = 80


    # if i == 4:
    #     range_min_i = 70
    #     range_max_i = 100
    # if i > 6:
    #     range_min_i = 70
    #     range_max_i = 100


    x_min = np.argmin(psd_diff[range_min_i:range_max_i]) + range_min_i
    first_notch_index[i] = x_min
    ax3.arrow(f_l[x_min], psd_diff[x_min]+dist_between_plots*i-2*dist_between_plots, 0, dist_between_plots, head_width=110, head_length=dist_between_plots, fc='k', ec='k')

    # # find second notch
    # range_min_i = 50
    # range_max_i = 70
    # x_min = np.argmin(psd_diff[range_min_i:range_max_i]) + range_min_i
    # second_notch_index[i] = x_min






# change y tick  labels to 0
ax3.set_yticks(np.arange(int(len(wav_files_only)/2))*dist_between_plots)
labels = [ '0' for item in np.arange(int(len(wav_files_only)/2))]
ax3.set_yticklabels(labels)
ax3.set_title(splitted[0]+' '+splitted[3])
# Shrink current axis by 20%
box = ax3.get_position()
ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax3.set_xticks(range(0,15000,1000))
ax3.set_xticklabels(range(0,15),fontweight='bold')
ax3.set_yticklabels([])
# plt.savefig("test.svg")

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(4,1,1)
ax.plot(first_notch_index)
ax.set_title('First Notch Movement')
ax.set_ylabel('Notch Frequency Index')
ax.set_xticks(np.arange(int(len(wav_files_only)/2)))
labels = [ item.split('_')[6+number_separators] for item in wav_files_only[::2]]
ax.set_xticklabels(labels)




#
#
# ax = fig.add_subplot(4,1,2)
# ax.plot(second_notch_index)
# ax.set_title('Second Notch Movement')
# ax.set_ylabel('Notch Frequency Index')


plt.show()


