import pygame
import threading
import queue
from openbci import OpenBCICyton
import numpy as np
import joblib
from signal_processing import bandpass_filter, extract_features

WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EEG-Controlled Square")

q = queue.Queue()
model = joblib.load('models/eeg_model.pkl')

def eeg_thread(port='COM3'):
    def callback(sample):
        raw = np.array(sample.channels_data)
        filtered = bandpass_filter(raw)
        features = extract_features(filtered).reshape(1, -1)
        prediction = model.predict(features)[0]
        q.put(prediction)
    board = OpenBCICyton(port=port)
    board.start_stream(callback)

def main():
    pygame.init()
    clock = pygame.time.Clock()
    x, y = WIDTH // 2, HEIGHT // 2
    size = 50
    speed = 5
    threading.Thread(target=eeg_thread, daemon=True).start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not q.empty():
            cmd = q.get()
            if cmd == 'focus':
                x += speed
            elif cmd == 'relax':
                x -= speed
            elif cmd == 'blink':
                y += speed

        screen.fill((30, 30, 30))
        pygame.draw.rect(screen, (0, 255, 0), (x, y, size, size))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()

