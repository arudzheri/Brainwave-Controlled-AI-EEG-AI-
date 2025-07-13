from openbci import OpenBCICyton
import numpy as np
from signal_processing import bandpass_filter

def handle_sample(sample, callback):
    # Get raw channel data
    raw = np.array(sample.channels_data)
    # Filter signal
    filtered = bandpass_filter(raw)
    # Callback with filtered data
    callback(filtered)

def start_stream(callback, port='COM3'):
    board = OpenBCICyton(port=port)
    board.start_stream(lambda sample: handle_sample(sample, callback))

