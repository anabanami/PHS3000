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
K_1 = 1
Sn = 1

# Desintegration energy
# Cs-137 disintegrates by beta minus emission to the excited state of Ba-137 (94.6 %)
theory_T = 0.5120 * MeV
theory_T_rel = theory_T / rel_energy_unit
theory_w_0_rel =  theory_T_rel + 1
p_0_rel = np.sqrt(theory_w_0_rel**2 - 1)

data = spa.betaray.read_data(r'beta-ray_data.csv')

def csv(data_file):
    # valid data slicing from csv file
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

    # make np array
    lens_current = np.array(lens_current)
    return background_count_data, count, lens_current, u_lens_current

def k(lens_current):
    # Finding constant of proportionality in p = kI
    # calibration peak (K) index of k peak is i=20
    T_K = 624.21 * keV / rel_energy_unit
    I_k = lens_current[20] 
    k = np.sqrt((T_K + 1)**2 - 1) / I_k

    # defining an appropriate uncertainty for our k peak
    u_I_k = 0.1 / 2
    u_k = k * (u_I_k / lens_current[20])
    # print(f"\nabsolute uncertainty in I_k = {u_I_k:.3f}")
    # print(f"absolute uncertainty in k = {u_k:.3f}")
    # print(f"fractional uncertainty in k = {u_k / k:.3f}\n")
    return k, u_k

def p_rel(lens_current, k, u_k):
    # The momentum spectrum (relativistic units)
    p_rel = k * lens_current
    # uncertainty in p
    u_p_rel = p_rel * np.sqrt((u_k / k)**2 + (0.0005 / lens_current)**2)
    # print(f"absolute uncertainty u(p_rel):\n {u_p_rel}")
    # print(f"fractional uncertainty u(p_rel) / p_rel:\n {(u_p_rel / p_rel)}")
    dp_rel = p_rel[1]-p_rel[0]
    return p_rel, u_p_rel, dp_rel

def interpolated_fermi(p_rel):
    # KURIE/Fermi PLOT
    fermi_data = spa.betaray.modified_fermi_function_data
    return interp1d(fermi_data[:,0], fermi_data[:,1], kind='cubic')(p_rel)

def n(p_rel):
    w_rel = np.sqrt(p_rel**2 + 1) # relativistic energy units
    u_w_rel = u_p_rel[8:18]
    n = K_1 * Sn * (w_rel * interpolated_fermi(p_rel) / p_rel) * p_rel**2 * (theory_w_0_rel - w_rel)**2
    return w_rel, u_w_rel

def correct_count(background_count_data):
    # correcting our data by removing avg background count
    background_count = []
    for row in background_count_data:
        background_count.append(row[5])
    avg_background_count = np.mean(background_count)
    # print(f"We want to subtract the background count from our data {avg_background_count=}")
    # calculating fractional uncertainty in total background count (delta_t = 24 min)
    total_background = np.sum(background_count)
    u_avg_background_count = np.sqrt(total_background) / 4

    # uncertainty in the corrected count
    background_corrected_count = count - avg_background_count

    ##################################################################################
    # I chose the uncertainty in the count to be 10 counts
    u_background_corrected_count = np.sqrt(10**2 + u_avg_background_count**2)
    ##################################################################################

    # As per Siegbahn [9] correction for spectrometers resolution
    correct_count = background_corrected_count / lens_current
    u_correct_count = np.sqrt((u_background_corrected_count / background_corrected_count)**2 + (u_lens_current / lens_current)**2)
    return correct_count, u_correct_count

def f(x, m, c):
    # linear model for optimize.curve_fit()
    return m * x + c

# shape factor from Siegbahn
def S_n(x, opt_w_n, u_opt_w_n):
    u_S_n = np.sqrt((2 * u_x * x)**2 + (2 * np.sqrt(u_opt_w_n**2 + u_x**2) * (opt_w_n - x))**2)
    S_n = x**2 - 1 + (opt_w_n - x)**2
    return S_n, u_S_n

def LHS(S_n):
    y = np.sqrt(correct_count[8:18] / (p_rel[8:18] * x * interpolated_fermi(p_rel[8:18]) * S_n))
    u_y = (y / 2) * np.sqrt((u_correct_count[8:18] / correct_count[8:18])**2 + (2 * (u_p_rel[8:18] / p_rel[8:18])**2) + (u_interpolated_fermi / interpolated_fermi(p_rel[8:18]))**2 + (u_S_n / S_n)**2)
    return y, u_y

def optimal_fit(f, x, y, u_y):
    # unpack into popt, pcov
    popt, pcov = scipy.optimize.curve_fit(f, x, y, sigma=u_y, absolute_sigma=False)
    # To compute one standard deviation errors on the parameters use 
    perr = np.sqrt(np.diag(pcov))

    # optimal parameters
    opt_K_2, opt_intercept = popt
    u_opt_K_2, u_opt_intercept = perr
    print(f"\noptimised gradient {opt_K_2:.3f} ± {u_opt_K_2:.3f}")
    print(f"optimised intercept {opt_intercept:.3f} ± {u_opt_intercept:.3f}")

    optimised_fit = f(x, opt_K_2, opt_intercept)
    # uncertainty in linear model f given optimal fit
    u_f = np.sqrt((x * u_opt_K_2)**2 + (u_opt_intercept)**2)
    # return optimal parameters
    return opt_K_2, opt_intercept, u_opt_K_2, u_opt_intercept

# comparison to theory
def compare(opt_w_n, u_opt_w_n):
# using our results to find T
    opt_T_rel = opt_w_n - 1
    u_opt_T_rel = opt_T_rel * (u_opt_w_n / opt_w_n)
    opt_T = opt_T_rel * rel_energy_unit / MeV
    u_opt_T = u_opt_T_rel * (rel_energy_unit / MeV)
    diff = 0.512 - opt_T
    how_many_sigmas = diff / u_opt_T
    print(f"\nEXPECTED RESULT T = {theory_T / MeV :.3f} MeV")
    print(f"(optimised) T = {opt_T:.3f} ± {u_opt_T:.3f} MeV")
    # print(f"difference {diff:.3f}")
    print(f"number of 𝞼: {- how_many_sigmas:.3f}")

########################### Calling our functions ###########################

# open, read and dissect data file
background_count_data, count, lens_current, u_lens_current = csv(data)

# find constant k
k, u_k = k(lens_current)

# find momentum spectrum
p_rel, u_p_rel, dp_rel = p_rel(lens_current, k, u_k)

# correct background count (accounting for background and resolution (3%))
correct_count, u_correct_count = correct_count(background_count_data)

######################### Linear fit ##########################
# our sliced data linearised
x, u_x = n(p_rel[8:18])

# uncertainty in interpolated fermi
u_interpolated_fermi = np.sqrt((u_p_rel[8:18] / p_rel[8:18])**2 + (u_x / x)**2) * interpolated_fermi(p_rel[8:18])

# LINEARISED KURIE WITH RESOLUTION CORRECTION
y = np.sqrt(correct_count[8:18] / (p_rel[8:18] * x * interpolated_fermi(p_rel[8:18])))
u_y = (y / 2) * np.sqrt((u_correct_count[8:18] / correct_count[8:18].clip(min=1))**2 + (2 * (u_p_rel[8:18] / p_rel[8:18])**2) + (u_interpolated_fermi / interpolated_fermi(p_rel[8:18]))**2)

# first fit
opt_K_2, opt_intercept, u_opt_K_2, u_opt_intercept = optimal_fit(f, x, y, u_y)
# using our parameters to find opt_w_0
opt_w_0 = opt_intercept / - opt_K_2
u_opt_w_0 = np.sqrt((u_opt_K_2 / opt_K_2)**2 + (u_opt_intercept / opt_intercept)**2) * opt_w_0

# ANALYSIS using Shape factor
# shape factor
S_n, u_S_n = S_n(x, opt_w_0, u_opt_w_0)
yn, u_yn = LHS(S_n)
opt_K_2, opt_intercept, u_opt_K_2, u_opt_intercept = optimal_fit(f, x, yn, u_yn)

# using our results to find opt_w_1
opt_w_1 = opt_intercept / - opt_K_2
u_opt_w_1 = np.sqrt((u_opt_K_2 / opt_K_2)**2 + (u_opt_intercept / opt_intercept)**2) * opt_w_1

compare(opt_w_1, u_opt_w_1)

############################ plots ############################
# # OPTIMISED FIT PLOT
# plt.figure()
# plt.errorbar(
#             x, y, xerr=u_p_rel[8:18], yerr=u_y,
#             marker="None", linestyle="None", ecolor="m", 
#             label=r"$y = (\frac{n}{p w G})^{\frac{1}{2}}$", color="g", barsabove=True
# )
# plt.plot(
#         x, optimised_fit, marker="None",
#         linestyle="-", 
#         label="linear fit"
# )
# plt.fill_between(
#                 x, optimised_fit - u_f,
#                 optimised_fit + u_f,
#                 alpha=0.5,
#                 label="uncertainty in linear fit"
# )
# plt.title("Optimised linear fit for Kurie data")
# plt.xlabel(r"$w [mc^{2}]$")
# plt.ylabel(r"$\left ( \frac{n}{p w G} \right )^{\frac{1}{2}}$", rotation=0, labelpad=18)
# plt.legend()
# # spa.savefig('OPTIMISED_Kurie_linear_data_plot_.png')
# # plt.show()

# optimised_residuals = optimised_fit - y
# # plot
# plt.figure()
# plt.errorbar(
#             x, optimised_residuals, xerr=u_p_rel[8:18], yerr=u_f,
#             marker="o", ecolor="m", linestyle="None",
#             label="Residuals (linearised data)"
# )
# plt.plot([x[0], x[-1]], [0,0], color="k")
# plt.title("Residuals: optimised fit for linear Kurie data")
# plt.xlabel(r"$w [mc^{2}]$")
# plt.ylabel(r"$\left ( \frac{n}{p w G} \right )^{\frac{1}{2}}$", rotation=0, labelpad=18)
# plt.legend()
# # spa.savefig('OPTIMISED_linear_residuals_Kurie_linear_data.png')
# # plt.show()
