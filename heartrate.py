from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
# Calculate heart rate from FFT peaks
def find_heart_rate(fft, freqs, freq_min, freq_max):
    fft_maximums = []

    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    peaks, properties = signal.find_peaks(fft_maximums)
    max_peak = -1
    max_freq = 0
    res = [ele for ele in freqs if ele >=0]
    leng=len(freqs)-len(res)
    fft_maximums = fft_maximums[:-leng]
    # print("freqs", res)
    # print("fft_max", fft_maximums)

    plt.plot(res,fft_maximums, "r")
    plt.xticks(range(0,int(max(res)),1))
    plt.yticks(np.arange(0.1,2, step=0.2))

    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()
    # print("peaks:", peaks)
    # Find frequency with max amplitude in peaks
    for peak in peaks:
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak

    # print("max_freq", max_freq)
    #
    # print("max_peak", max_peak)
    return freqs[max_peak] * 60
