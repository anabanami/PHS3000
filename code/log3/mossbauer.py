# PHS3000
# Mossbauer effect
# Ana Fabela, 21/09/2020
import os
from pathlib import Path
import monashspa.PHS3000 as spa
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

plt.rcParams['figure.dpi'] = 150
folder = Path('spectra')
os.makedirs(folder, exist_ok=True)

# Globals
E = np.linspace(1, 512,512)

hbar = 1.0545718e-34 # [Js]
c = 299792458 # [m/s]
mass_e = 9.10938356e-31 # [kg]
eV = 1.602176634e-19 # [J]
meV = 1e-3 * eV
keV = 1e3 * eV

rel_energy_unit = mass_e * c**2 

E𝛾 = 14.4 * keV # [J]
amu = 1.6605402e-27 # [kg]
Fe_A = 55.845 * amu # [kg]
u_Fe_A = 2e-3 * amu # [kg]
# print(f"atomic weight of Fe atom : ({Fe_A*1e26:.4f} ± {u_Fe_A*1e26:.4f}) ×10^26 kg")

def recoil_energy():
    # recoil energy of a single Fe atom
    E_R = E𝛾**2 / (2 * Fe_A * c**2)
    # print(f"\nE𝛾**2= {E𝛾**2}keV^2, {Fe_A =}kg, {c**2 =}(m/s)^2")
    print(f"\nE_R : {E_R / meV} meV")
    return E_R

def read_files():
    # read data files and return numpy arrays
    data_KFe = np.array([int(value) for value in open('KFe.txt')])
    data_αFe = np.array([int(value) for value in open('alpha_Fe.txt')])
    return data_KFe, data_αFe

def plot_data(E, data_file, fit):
    plt.plot(E, data_file, linestyle="None", marker='.', color='plum',label=r"detected $\gamma-$rays")
    plt.plot(E, fit, color='teal',label=r"fit")
    plt.xlabel('Bins')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()

def not_a_Lorentzian(E, B, A, E_0, Γ):
    return B - (A * (Γ / 2)**2 /((E - E_0)**2 + (Γ / 2)**2))

def not_6_Lorentzians(E, B, A_1, E_0_1, Γ_1, A_2, E_0_2, Γ_2, A_3, E_0_3, Γ_3, A_4, E_0_4, Γ_4, A_5, E_0_5, Γ_5, A_6, E_0_6, Γ_6):
    return B -( A_1 * (Γ_1 / 2)**2 /((E - E_0_1)**2 + (Γ_1 / 2)**2) +\
    A_2 * (Γ_2 / 2)**2 /((E - E_0_2)**2 + (Γ_2 / 2)**2) +\
    A_3 * (Γ_3 / 2)**2 /((E - E_0_3)**2 + (Γ_3 / 2)**2) +\
    A_4 * (Γ_4 / 2)**2 /((E - E_0_4)**2 + (Γ_4 / 2)**2) +\
    A_5 * (Γ_5 / 2)**2 /((E - E_0_5)**2 + (Γ_5 / 2)**2) +\
    A_6 * (Γ_6 / 2)**2 /((E - E_0_6)**2 + (Γ_6 / 2)**2) )

def KFe_spectrum_fitting(E, data_KFe):
    # Potassium ferrocyanide data
    # initial parameters from eyeballing plot
    initial_guess = [395 * 1e3, 60 * 1e3, 234, 80]
    pars, pcov = scipy.optimize.curve_fit(not_a_Lorentzian, E, data_KFe, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    fit = not_a_Lorentzian(E, *pars)
    # return fit parameters
    return pars, fit

def αFe_spectrum_fitting(E, data_αFe):
    # α-Iron data
    # initial parameters from eyeballing plot
    initial_guess = [560 * 1e3, 60 * 1e3, 90, 20, 50 * 1e3, 160, 20, 30 * 1e3, 225, 20, 30 * 1e3, 280, 20, 50 * 1e3, 345, 20, 60 * 1e3, 410, 20]
    pars_6, pcov = scipy.optimize.curve_fit(not_6_Lorentzians, E, data_αFe, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    fit_6 = not_6_Lorentzians(E, *pars_6)
    print(f"\n{pars_6=}\n{perr=}")
    # return fit parameters
    return pars_6, fit_6

#*~ function calls ~*#
data_KFe, data_αFe = read_files()
pars, fit = KFe_spectrum_fitting(E, data_KFe)
# plot_data(E, data_KFe, fit)

pars_6, fit_6 = αFe_spectrum_fitting(E, data_αFe)
# plot_data(E, data_αFe, fit_6)


