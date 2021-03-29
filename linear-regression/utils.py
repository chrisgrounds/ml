import sys
import matplotlib.pyplot as plt
import numpy as np


def show_output(input_data, score, prediction):
    print(f"Number to increment: {input_data}")
    print(f"Model accuracy: {score}")
    print(f"Model prediction: {prediction}")

    plt.plot(input_data, prediction)
    plt.show()


def get_input_data():
    in_1, in_2 = float(sys.argv[1]), float(sys.argv[2])

    return np.arange(in_1, in_2).reshape(-1, 1)
