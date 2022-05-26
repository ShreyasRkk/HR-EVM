from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
# Calculate heart rate from FFT peaks
def find_heart_rate(fft, freqs, freq_min, freq_max):
    fft_maximums = []
    print("fft.shape[0]", fft.shape)
    for i in range(fft.shape[0]): #fft.shape[0]= number of frames
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            # print("fftMap", fftMap)
            # print("fftMap", fftMap.max())
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    peaks, properties = signal.find_peaks(fft_maximums)
    print("peaks", )
    print("shape of fft_max", np.shape(fft_maximums))
    max_peak = -1
    max_freq = 0
    res = [ele for ele in freqs if ele >=0]
    leng=len(freqs)-len(res)
    fft_maximums = fft_maximums[:-leng]
    # print("freqs", res)
    # print("fft_max", fft_maximums)

    plt.plot(res,fft_maximums, "r")
    plt.xticks(np.arange(0,5,step=1))
    plt.yticks(np.arange(0.1,2, step=0.2))

    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()
    # print("peaks:", peaks)
    # Find frequency with max amplitude in peaks
    for peak in peaks:
        print("freqs[peak]", freqs[peak])
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak

    # print("max_freq", max_freq)
    # print("max_peak", max_peak)
    return freqs[max_peak] * 60
