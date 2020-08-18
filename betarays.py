import monashspa.PHS3000 as spa
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
plt.rcParams['figure.dpi'] = 150

# Globals
c = 299792458 # [m/s]
mass_e = 9.10938356e-31 # [kg]
eV = 1.602176634e-19 # [J]
MeV = 1e6 * eV
keV = 1e3 * eV
rel_energy_unit = mass_e * c**2 # to convert SI into relativistic or viceversa

data = spa.betaray.read_data(r'beta-ray_data.csv')

#valid data slicing
j = 0
for row in data:
    # print(f'{j=}')
    if row[0] == pd.Timestamp('2020-08-18 10:26:00+10:00', tz=pytz.FixedOffset(360)):
        valid_data = data[j:]
        continue
    j+=1

background_count_data = []
count = []
lens_current = []
u_lens_current = []
for row in valid_data:
    if row[3] == 'Closed':
        # print(row[3])
        background_count_data.append(row)
        continue
    count.append(row[5])
    lens_current.append(row[6])
    u_lens_current.append(row[7])

background_count = []
# correcting our data by removing avg background count
for row in background_count_data:
    background_count.append(row[5])
avg_background_count = np.mean(background_count)
# print(f"We want to substract thie background count from our data {avg_background_count=}")
# calculating fractional uncertainty in total background count (delta_t = 24 min)
total_background = np.sum(background_count)
u_avg_background_count = np.sqrt(total_background) / 4

# Finding constant of proportionality in p = kI
#calibration peak (K) index of k peak is i=20
T_K = 624.21 * keV / rel_energy_unit
k = np.sqrt((T_K + 1)**2 - 1) / lens_current[20]
print(f"{k=}")

# # plot
# plt.plot(
#         lens_current, count, marker="None",
#         linestyle="-", 
#         label="N(I)"
# )
plt.plot(
        lens_current, count - avg_background_count, marker="None",
        linestyle="-", 
        label="N(I) - background"
)
plt.title(r"$\beta^{-}$ particle count (corrected) as a function of lens coil current")
plt.xlabel("I [A]")
plt.ylabel("N(I) [arb]")
plt.legend()
spa.savefig('count_vs_lens_current_no_background.png')
# plt.show()




# pprint(f'{sorted(count)=}')
# pprint(f'{lens_current=}')
