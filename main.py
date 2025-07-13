from openbci import OpenBCICyton
import numpy as np
import joblib
from signal_processing import bandpass_filter, extract_features

model = joblib.load('models/eeg_model.pkl')

def handle_sample(sample):
    raw = np.array(sample.channels_data)
    filtered = bandpass_filter(raw)
    features = extract_features(filtered).reshape(1, -1)
    prediction = model.predict(features)[0]
    print(f"Brain Command: {prediction}")
    if prediction == 'focus':
        print("🧠 Trigger: Start work task")
    elif prediction == 'relax':
        print("🌙 Trigger: Play relaxation music")
    elif prediction == 'blink':
        print("👁️ Trigger: Pause app")

def main(port='COM3'):
    board = OpenBCICyton(port=port)
    board.start_stream(handle_sample)

if __name__ == "__main__":
    main()

