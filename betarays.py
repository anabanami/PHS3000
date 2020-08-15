import monashspa.PHS3000 as spa
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

data = spa.betaray.read_data(r'betaray_data2.csv')

# GLOBALS
# physical constants and SI units
hbar = 1.0545718e-34 # [Js]
c = 299792458 # [m/s]
epsilon_0 = 8.854e-12 
mass_e = 9.10938356e-31 # [kg]
e_charge = 1.60217662e-19 # [C]
eV = 1.602176634e-19 # [J]
meV = 1e-3 * eV
keV = 1e6 * eV
angstrom = 1e-10 # [m]


T_K = 624.21 * keV

w = T + mass_e * c**2

# i = 0
# for run in data:
#     # print(f'{i=}')
#     if run[0] == pd.Timestamp('2020-08-06 13:39:36+10:00', tz=pytz.FixedOffset(600)):
#         valid_data = data[i:]
#         continue
#     i+=1

count = []
lens_current = []
for row in data:
    count.append(row[5])
    lens_current.append(row[6])

plt.figure(2)
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
spa.savefig('betarays_fig2.png')
plt.show()

# print(f'{count=}')
# print(f'{lens_current=}')
