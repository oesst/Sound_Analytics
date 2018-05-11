import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import soundfile as sf
from matplotlib.font_manager import FontProperties
import pyaudio
from scipy.signal import welch, butter, lfilter, savgol_filter, group_delay
from os import listdir
from os.path import isfile, join
from colour import Color

# This script displays spectral cues extracted from several wav files

plt.style.use('ggplot')

CHUNK_SIZE = 4096
RATE = 44100
FORMAT = pyaudio.paInt16

# path = '/home/oesst/ownCloud/PhD/binaural head/recordings/full_head/whiteNoise_1_20000Hz_simplePinna/azimuth_0/'
file_left = '/home/oesst/ownCloud/PhD/Code/Python/sound_analyzer/recordings/full_head/whiteNoise_1_20000Hz_normalEars/azimuth_0/whiteNoise_1_20000Hz_normalEars_azi_0_ele_0_left.wav'
file_right = '/home/oesst/ownCloud/PhD/Code/Python/sound_analyzer/recordings/full_head/whiteNoise_1_20000Hz_normalEars/azimuth_0/whiteNoise_1_20000Hz_normalEars_azi_0_ele_0_right.wav'

dist_between_plots = 6
number_separators = 1


def get_welch_spectrum(data):
    f, psd = welch(data, RATE, nperseg=128)
    return f, psd


print("Opening files %s and %s" % (file_right, file_left))

data_l = sf.read(file_right)[0]
data_r = sf.read(file_left)[0]



# # get welch spectrum
f_l, psd_l = get_welch_spectrum(data_l)
f_r, psd_r = get_welch_spectrum(data_r)

# foo = group_delay(psd_l)

# make log scale
psd_l = np.log10(psd_l) * 20
psd_r = np.log10(psd_r) * 20

# cut off frequencies at the end (>18000Hz)
indis = f_l < 15000

f_l = f_l[indis]
f_r = f_r[indis]
psd_l = psd_l[indis]
psd_r = psd_r[indis]


psd_l_deriv = -np.diff(psd_l)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(2, 1, 1)
ax.plot(psd_l)
ax = fig.add_subplot(2, 1, 2)
ax.plot(psd_l_deriv)



plt.show()
