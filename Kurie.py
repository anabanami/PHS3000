# PHS3000 - LOGBOOK1
# Beta decay - Momentum spectrum 
# Ana Fabela, 16/08/2020

import monashspa.PHS3000 as spa
from scipy.interpolate import interp1d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

#fake momentum space
p_rel = np.linspace(1e-6, 3, 100) # relativistic units [mc]
dp_rel = p_rel[1]-p_rel[0]

# getting interpolated
fermi_data = spa.betaray.modified_fermi_function_data
interpolated_fermi = interp1d(fermi_data[:,0], fermi_data[:,1], kind='cubic')

# Globals
c = 299792458 # [m/s]
mass_e = 9.10938356e-31 # [kg]
eV = 1.602176634e-19 # [J]
MeV = 1e6 * eV
keV = 1e3 * eV
# UNITS WTF
rel_energy_unit = mass_e * c**2 # to convert SI into relativistic or viceversa
# w = np.sqrt(p_rel**2 * c**2 + mass_e**2 * c**4) # [SI]

# Desintegration energy
# Cs-137 disintegrates by beta minus emission to the ground state of Ba-137 (5,6 %)
w_0 = 1.174 * MeV
w_0_rel = w_0 / rel_energy_unit
p_0_rel = np.sqrt(w_0_rel**2 - 1) / (mass_e * c)
print(p_0_rel)

K_1 = 1 # ?
S_0 = 1
def n(p_rel):
    w_rel = np.sqrt(p_rel**2 + 1) # relativistic energy units
    n = K_1 * S_0* (w_rel* interpolated_fermi(p_rel) / p_rel) * p_rel**2 * (w_0_rel - w_rel)**2
    return n, w_rel

n_p, w_rel = n(p_rel) #  call and unpack n(p)
# equation (3) in lab script
N = n_p * dp_rel

# # plot
plt.plot(
        p_rel, N, marker="None",
        linestyle="-"
)
plt.title("fake news")
plt.xlabel("p [mc]")
plt.ylabel("n(p)dp")
spa.savefig('Kurie_plot_fake.png')
plt.show()

############################# Linear fit #############################

# n_p, w_rel = n(p_rel) #  call and unpack n(p)
x = w_rel
y = np.sqrt(n_p / (p_rel * w_rel * interpolated_fermi(p_rel)))
u_y = 0
linear_results = spa.linear_fit(x, y)#, u_y=u_A_lin)

# # making our linear fit with one sigma uncertainty
n_fit = linear_results.best_fit
u_n_fit = linear_results.eval_uncertainty(sigma=1)

# # plot
# plt.errorbar(
#             x, y, xerr=0, yerr=u_y,
#             marker="o", linestyle="None",
#             label="Activity data"
# )
plt.plot(
        x, n_fit, marker="None",
        linestyle="-", 
        label="linear fit"
)
plt.fill_between(
                x, n_fit - u_n_fit,
                n_fit + u_n_fit,
                alpha=0.5,
                label="uncertainty in linear fit"
)
plt.title("Linearised Kurie plot")
plt.xlabel(r"$w$ [mc]")
plt.ylabel(r"$\left ( \frac{n}{p w G} \right )^{\frac{1}{2}}$", rotation=0, labelpad=18)
plt.legend()
spa.savefig('Kurie_linear_plot_fake.png')
plt.show()

############################# Linear fit #############################