import os
from pathlib import Path
import monashspa.PHS3000 as spa
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

plt.rcParams['figure.dpi'] = 150

x = np.linspace(0,1036,1024) # bins array

def read_files():
    read_folder = Path('log2')
    files = list(os.listdir(path=read_folder))
    data_files = []
    p_values = []
    files.sort(key=lambda name: int(name.split('_')[0]) if name[0].isdigit() else -1)
    for i, file in enumerate(files):
        # print(i, file)
        if i >= 1:
            p_value = float(file.rstrip('mbar.mca').replace('_', '.'))
            p_values.append(p_value)
        header, data = spa.read_mca_file(read_folder/file)
        data_files.append(data)
    return p_values, data_files


def plot_data(x, y, i, average, ax=None):
    average = [average] * 41

    xmax = np.argmax(y)
    ymax = y.max()
    # half_y_max = ymax / 2
    half_y_max = np.ceil(ymax / 2)
    L = np.argmin((y[:xmax] - half_y_max)**2)
    R = np.argmin((y[xmax:] - half_y_max)**2) + xmax

    print(f"{xmax=}")
    print(f"{ymax=}")
    print(f"{half_y_max=}\n")
    print(f"{L=}")
    print(f"{R=}")
    peak_width = R - L
    print(f"{peak_width=}")


    plt.bar(x, y, color='tomato',label="Detected alphas")
    plt.plot(x[:41], average, label="Extrapolation") # extrapolation cuve
    # plt.axhline(y=half_y_max, linestyle=':', alpha=0.3, label="Half max")
    # plt.fill_betweenx([0, ymax + 20], [L, L], [R, R], alpha=0.3, zorder=10)

    plt.show()
    plt.clf()
    return xmax, ymax, peak_width

def energy_peaks(p_values, data_files):
    max_counts = []
    max_positions = []
    peak_widths = []
    total_events = []

    for i, signal in enumerate(data_files):
        if i<1:
            continue
        #  sum of all events
        total = np.sum(signal)
        total_events.append(total)
        # curve extrapolation for the threshold region
        average = np.mean(signal[42:79])

        # barcharts to visualise our files
        xmax, countmax, peak_width = plot_data(x, signal, i, average)

        max_positions.append(xmax)
        max_counts.append(countmax)
        peak_widths.append(peak_width)
    peak = max_positions[1]
    return peak, signal, total_events, max_positions, max_counts, peak_widths


p_values, data_files = read_files()

peak, signal, total_events, max_positions, max_counts, peak_widths = energy_peaks(p_values, data_files)