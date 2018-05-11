import wave
import numpy as np
import struct
import soundfile as sf
import matplotlib.pyplot as plt

signal_r ='/home/oesst/ownCloud/PhD/Sounds/sinus_500Hz_wn_bg.wav'
signal_l = '/home/oesst/ownCloud/PhD/Sounds/sinus_500Hz_different_wn_bg_quarter_phase_shifted.wav'


def gcc(a, b,max_delay = 0):
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


def cross_correlation_using_fft( x, y):
    from numpy.fft import fft, ifft,     fftshift

    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)


def find_delay( a, b, max_delay=0):
    # very accurate but not so fast as gcc
    # from scipy.signal import correlate
    # corr = correlate(a, b)
    # corr = np.correlate(a,b,'full')
    corr = cross_correlation_using_fft(a, b)
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



def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc


data_r= sf.read(signal_r)
data_l= sf.read(signal_l)

data_r = data_r[0]
data_l = data_l[0]



# delay,gphat = gcc_phat(data_r,data_l,fs=44100)
#
# print(delay/44100 *1000)
#
# delay_1,gphat_1 = gcc(data_r,data_l,352)
#
# print(delay_1)
# print(delay_1/44100*1000)

delay_2 = find_delay(data_r,data_l,50)


print(delay_2)
print(delay_2/44100*1000)







# [xr,lag]=xcorr(signal_l,signal_r);
#
# [mx,mind]=max(abs(xr));
#
# delay_eva=lag(mind)


# print((len(tmp) / 2.0 - lag)/44100 *1000)
#
#print(delay)
# print(delay_1)
#
plt.plot(data_r[1:20000])
plt.plot(data_l[1:20000])
plt.show()
