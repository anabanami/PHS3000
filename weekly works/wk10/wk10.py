# PHS3000
# week 10 - FOURIER TRANSFORMS
# Ana Fabela Hinojosa, 23/10/2020
import numpy as np
import matplotlib.pyplot as plt

nterms = 250 # modify this value to get desired number of sinusoidal components

t = np.linspace(0, 2 * np.pi, 1024)
omega = np.arange(1, 2*nterms, 2).reshape(nterms,1)

# print(f"{t}")
# print(f"{omega}")

sin = np.sin(omega * t)
# print(f"{sin}")

S = (4 / (np.pi * omega) * sin).sum(axis=0)
# print(f"{S}")

plt.plot(t, S)
plt.show()