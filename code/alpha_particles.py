# PHS3000
# alpha particles - Range and energy loss
# Ana Fabela, 09/09/2020
import os
from pathlib import Path
import monashspa.PHS3000 as spa
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

plt.rcParams['figure.dpi'] = 120

folder = Path('log2')
files = list(os.listdir(path=folder))
data_files = []

files.sort(key=lambda name: int(name.split('_')[0]) if name[0].isdigit() else -1)

for file in files:
    # print(file)
    header, data = spa.read_mca_file(folder/file)
    data_files.append(data)

# MAKING HISTOGRAMS
# number of bins
# K = 1024
# for i, file in enumerate(data_files):
#     # print(file)
#     counts, bin_edges = np.histogram(file, bins=K)
#     # # print(f"{counts = }, length of list: {len(counts)}")
#     # # print(f"{bin_edges[:-1]=}, length of list: {len(bin_edges[:-1])}")

#     # # plt.grid(linestyle=':')
#     plt.hist(data_files[1], bins=K, alpha=0.5)
#     plt.xlabel('Energy')
#     plt.ylabel('Counts')
#     # spa.savefig(f'file{{{i}}}.png')
#     plt.title(f'file[{i}].mca')
#     plt.show()