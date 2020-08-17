import monashspa.PHS3000 as spa
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

# Globals
c = 299792458 # [m/s]
mass_e = 9.10938356e-31 # [kg]
eV = 1.602176634e-19 # [J]
MeV = 1e6 * eV
keV = 1e3 * eV
#calibration peak (K)
# T_K = 624.21 * keV
# w = T_K + mass_e * c**2

data = spa.betaray.read_data(r'betaray_data.csv')

i = 0
for run in data:
    # print(f'{i=}')
    if run[0] == pd.Timestamp('2020-08-06 13:39:36+10:00', tz=pytz.FixedOffset(600)):
        valid_data = data[i:]
        continue
    i+=1

count = []
lens_current = []
for row in valid_data:
    count.append(row[5])
    lens_current.append(row[6])

# # plot
plt.plot(
        lens_current, count, marker="None",
        linestyle="-", 
        label="N(I)"
)
plt.title(r"$\beta^{-}$ particle count as a function of lens coil current")
plt.xlabel("I [A]")
plt.ylabel("N(I) [arb]")
plt.legend()
spa.savefig('count_vs_lens_current.png')
plt.show()

print(f'{count=}')
print(f'{lens_current=}')
