import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut=1.0, highcut=50.0, fs=250, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=250, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_features(signal):
    # Example features: mean, std, max, min
    return np.array([np.mean(signal), np.std(signal), np.max(signal), np.min(signal)])

