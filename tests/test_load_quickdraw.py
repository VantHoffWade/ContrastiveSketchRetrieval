import numpy as np
import matplotlib.pyplot as plt

def draw_sample(sketch):
    x, y = 0, 0
    strokes = []
    current_stroke = []

    for dx, dy, p in sketch:
        x += dx
        y += dy
        current_stroke.append((x, y))
        if p == 1:
            strokes.append(current_stroke)
            current_stroke = []
        if p == 2:
            if current_stroke:
                strokes.append(current_stroke)
            break

    for stroke in strokes:
        stroke = np.array(stroke)
        plt.plot(stroke[:, 0], -stroke[:, 1])  # Y轴反转
    plt.axis('equal')
    plt.show()

quickdraw_dataset_path = r"C:\Users\Administrator\Downloads\sketchrnn_The Eiffel Tower.full.npz"
quickdraw_dataset = np.load(quickdraw_dataset_path, encoding='latin1', allow_pickle=True)

draw_sample(quickdraw_dataset["train"][2])