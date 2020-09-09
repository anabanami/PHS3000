# PHS3000
# Betarays - radioactive decay Cs - 137
# Ana Fabela, 15/08/2020
import monashspa.PHS3000 as spa
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
import scipy.optimize

plt.rcParams['figure.dpi'] = 150

# Globals
hbar = 1.0545718e-34 # [Js]
c = 299792458 # [m/s]
mass_e = 9.10938356e-31 # [kg]
eV = 1.602176634e-19 # [J]
MeV = 1e6 * eV
keV = 1e3 * eV
rel_energy_unit = mass_e * c**2 # to convert SI into relativistic or viceversa


# Desintegration energy
# Cs-137 disintegrates by beta minus emission to the excited state of Ba-137 (94.6 %)
theory_T = 0.5120 * MeV
theory_T_rel = theory_T / rel_energy_unit
theory_w_0_rel =  theory_T_rel + 1
p_0_rel = np.sqrt(theory_w_0_rel**2 - 1)

data = spa.betaray.read_data(r'beta-ray_data.csv')

def csv(data_file):
    # extracting valid data from csv file
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

    # make lens current an np.array
    lens_current = np.array(lens_current)
    return background_count_data, count, lens_current, u_lens_current


def _1Lorentzian(x, amp1, cen1, wid1, c):
    return amp1 * wid1**2 /((x-cen1)**2 + wid1**2)


def _2Lorentzian(x, amp1, cen1, wid1, amp2, cen2, wid2, c):
    # To find constant of proportionality in p = kI
    # we fit calibration peak (K) with a lorentzian
    return (amp1 * wid1**2 / ((x-cen1)**2 + wid1**2)) + (amp2 * wid2**2 /((x-cen2)**2 + wid2**2)) + c

def k_fit(f, x, y):
    # 2 peak Lorentzian fit to find max of K-conversion peak
    # unpack into popt, pcov
    popt_2lorentz, pcov_2lorentz = scipy.optimize.curve_fit(_2Lorentzian, x, y, p0=[amp1, cen1, wid1, amp2, cen2, wid2, c])
    perr_2lorentz = np.sqrt(np.diag(pcov_2lorentz))
    pars_1 = popt_2lorentz[0:3]
    pars_2 = popt_2lorentz[3:6]
    lorentz_peak_1 = _1lorentzian(x, *pars_1)
    lorentz_peak_2 = _1lorentzian(x, *pars_2)
    # return fit parameters
    return pars_1, pars_2, lorentz_peak_1, lorentz_peak_2


# open, read and dissect data file
background_count_data, count, lens_current, u_lens_current = csv(data)

#first guess for lorentzian parameters
amp1, cen1, wid1, amp2, cen2, wid2, c = 1400, 0.9, 1.2, 1200, 1.8, 0.4, 170


plt.plot(lens_current,_2Lorentzian(lens_current, amp1, cen1, wid1, amp2, cen2, wid2, c))
plt.plot(lens_current, count)
plt.show()

pars_1, pars_2, lorentz_peak_1, lorentz_peak_2  = k_fit(_2Lorentzian(lens_current, amp1, cen1, wid1, amp2, cen2, wid2,c), lens_current, count)

plt.figure()
ax1.plot(lens_current, lorentz_peak_1, "g")
ax1.fill_between(lens_current, lorentz_peak_1.min(), lorentz_peak_1, facecolor="green", alpha=0.5)
  
ax1.plot(lens_current, lorentz_peak_2, "y")
ax1.fill_between(lens_current, lorentz_peak_2.min(), lorentz_peak_2, facecolor="yellow", alpha=0.5)  
# spa.savefig('lorentzian_fit.png')
plt.show()