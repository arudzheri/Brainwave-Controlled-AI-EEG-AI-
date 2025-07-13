import csv
from openbci import OpenBCICyton
from signal_processing import bandpass_filter
import numpy as np

def collect(label, port='COM3'):
    with open(f'data/{label}_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        def callback(sample):
            raw = np.array(sample.channels_data)
            filtered = bandpass_filter(raw)
            features = [label] + extract_features(filtered).tolist()
            writer.writerow(features)
        board = OpenBCICyton(port=port)
        print(f"Start collecting data for label '{label}'... Press Ctrl+C to stop.")
        try:
            board.start_stream(callback)
        except KeyboardInterrupt:
            print("Data collection stopped.")

def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
    ]

if __name__ == "__main__":
    label = input("Enter label for data collection (e.g. focus, relax): ")
    collect(label)

