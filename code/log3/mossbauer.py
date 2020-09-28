# PHS3000
# Mossbauer effect
# Ana Fabela, 28/09/2020
import os
from pathlib import Path
import monashspa.PHS3000 as spa
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

plt.rcParams['figure.dpi'] = 150

# folder = Path('spectra')
# os.makedirs(folder, exist_ok=True)

# Globals
hbar = 1.0545718e-34 # [Js]
c = 299792458 # [m/s]

mass_e = 9.10938356e-31 # [kg]
eV = 1.602176634e-19 # [J]
meV = 1e-3 * eV
keV = 1e3 * eV

rel_energy_unit = mass_e * c**2 

E𝛾 = 14.4 * keV # [J
amu = 1.6605402e-27 # [kg]
Fe_A = 55.845 * amu # [kg]
u_Fe_A = 2e-3 * amu # [kg]
# print(f"atomic weight of Fe atom : ({Fe_A*1e26:.4f} ± {u_Fe_A*1e26:.4f}) ×10^26 kg")

def read_files():
    # read data files and return numpy arrays
    data_KFe = np.array([int(value) for value in open('KFe.txt')])
    data_αFe = np.array([int(value) for value in open('alpha_Fe.txt')])
    return data_KFe, data_αFe

def spectrum_fitting(data_KFe, data_αFe):
    # for potassium ferrocyanide, the equation is:
    I_KFe = B − (A * (Γ / 2)**2 / ((E − E_0)**2 + (Γ**2 / 4)))
    # E is channel number, B is background count (baseline), 
    # E_0 is channel number for centre of the dip, A is depth 
    # of dip (in counts) and Γ is linewidth.
    I_αFe = B − (A * (Γ / 2)**2 / ((E − E_0)**2 + (Γ**2 / 4)))


def recoil_energy():
    # recoil energy of a single Fe atom
    E_R = E𝛾**2 / (2 * Fe_A * c**2)
    # print(f"\nE𝛾**2= {E𝛾**2}keV^2, {Fe_A =}kg, {c**2 =}(m/s)^2")
    print(f"\nE_R : {E_R / meV} meV")
    return E_R

#*~ function calls ~*#
data_KFe, data_αFe = read_files()
