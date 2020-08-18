# PHS3000 - LOGBOOK1
# Beta decay - Momentum spectrum 
# Ana Fabela, 15/08/2020
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

# uncertainty in the corrected count
corrected_count = count - avg_background_count
u_corrected_count = np.sqrt(count + u_avg_background_count**2)


# Finding constant of proportionality in p = kI
# calibration peak (K) index of k peak is i=20
T_K = 624.21 * keV / rel_energy_unit
k = np.sqrt((T_K + 1)**2 - 1) / lens_current[20]
# print(f"{k=}")
u_k = k * (0.0005 / lens_current[20]) 
# print(f"absolute uncertainty: {u_k = }")
# print(f"fractional uncertainty: {(u_k / k) = }\n")

# The momentum spectrum
lens_current = np.array(lens_current)
p_rel = k * lens_current
u_p_rel = p_rel * np.sqrt((u_k / k)**2 + (0.0005 / lens_current)**2)
# print(f"absolute uncertainty u(p_rel):\n {u_p_rel}")
# print(f"fractional uncertainty u(p_rel) / p_rel:\n {(u_p_rel / p_rel)}")
plt.figure(1)
plt.errorbar(
            p_rel, corrected_count, xerr=u_p_rel, yerr=u_corrected_count,
            marker="None",ecolor="m", label=r"$n(p)_{corrected}$", color="g", barsabove=True
)

plt.title(r"$\beta^{-}$ particle momentum spectrum")
plt.xlabel("p [mc]")
plt.ylabel("n(p)")
plt.legend()
spa.savefig('count_vs_momentum_no_background_error.png')
# plt.show()


########################KURIE PLOT########################
dp_rel = p_rel[1]-p_rel[0]

# getting interpolated
fermi_data = spa.betaray.modified_fermi_function_data
interpolated_fermi = interp1d(fermi_data[:,0], fermi_data[:,1], kind='cubic')

# Desintegration energy
# Cs-137 disintegrates by beta minus emission to the ground state of Ba-137 (5,6 %)
w_0 = 1.174 * MeV
w_0_rel = w_0 / rel_energy_unit
p_0_rel = np.sqrt(w_0_rel**2 - 1) / (mass_e * c)
# print(p_0_rel)

# #defining our count function
K_1 = 1 # ?
S_0 = 1
def n(p_rel):
    w_rel = np.sqrt(p_rel**2 + 1) # relativistic energy units
    n = K_1 * S_0* (w_rel* interpolated_fermi(p_rel) / p_rel) * p_rel**2 * (w_0_rel - w_rel)**2
    return n, w_rel

n_p_rel, w_rel = n(p_rel[:30]) #  call and unpack n(p)

# equation (3) in script
N = n_p_rel * dp_rel 

plt.figure(2)
plt.plot(
        p_rel[:30], N, marker="None",
        linestyle="-"
)
plt.title("Kurie relation")
plt.xlabel("p [mc]")
plt.ylabel("n(p)dp")
spa.savefig('Kurie_plot.png')
# plt.show()

############################# Linear fit #############################
y = np.sqrt(n_p_rel / (p_rel[:30]* w_rel * interpolated_fermi(p_rel[:30])))



u_y = 0




linear_results = spa.linear_fit(w_rel, y)#, u_y=u_A_lin)

# # making our linear fit with one sigma uncertainty
n_fit = linear_results.best_fit
u_n_fit = linear_results.eval_uncertainty(sigma=1)

plt.figure(3)
# # plot
plt.errorbar(
            w_rel, y, xerr=0, yerr=u_y,
            marker="o", linestyle="None",
            label="Activity data"
)
plt.plot(
        w_rel, n_fit, marker="None",
        linestyle="-", 
        label="linear fit"
)
plt.fill_between(
                w_rel, n_fit - u_n_fit,
                n_fit + u_n_fit,
                alpha=0.5,
                label="uncertainty in linear fit"
)
plt.title("Linearised Kurie plot")
plt.xlabel(r"$w$ [mc]")
plt.ylabel(r"$\left ( \frac{n}{p w G} \right )^{\frac{1}{2}}$", rotation=0, labelpad=18)
plt.legend()
spa.savefig('Kurie_linear_plot_.png')
plt.show()

############################# Linear fit #############################
##########################Linear fit residuals########################
# linear_residuals = A_lin_fit - A_lin # linear residuals (linear best fit - linearised data)

# plt.figure(4)
# # plot
# plt.errorbar(
#             time_lin, linear_residuals, xerr=0, yerr=u_A_lin,
#             marker="o", linestyle="None",
#             label="Residuals (linear fit-data)"
# )
# plt.plot([time[0], time[-1]], [0,0], color="k")
# plt.title("Figure 4: Residuals of linear fit of silver decay data")
# plt.xlabel("t [s]")
# plt.ylabel("ln(A) [arb]")
# plt.legend()
# spa.savefig('linear_residuals_silver_decay_data.png')
# plt.show()
# ##########################Linear fit residuals########################