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
# folder = Path('spectra')
# os.makedirs(folder, exist_ok=True)

# Globals
E = np.linspace(1, 512,512)
C_0 = 255.5
u_C_0 = 0.05

hbar = 1.0545718e-34 # [Js]
c = 299792458 # [m/s]
eV = 1.602176634e-19 # [J]
meV = 1e-3 * eV
keV = 1e3 * eV

E𝛾 = 14.4 * keV # [J]
u_E𝛾 = 0.05 * keV
amu = 1.6605402e-27 # [kg]
Fe_A = 55.845 * amu # [kg]
u_Fe_A = 2e-3 * amu # [kg]
# print(f"atomic weight of Fe atom : ({Fe_A*1e26:.4f} ± {u_Fe_A*1e26:.4f}) ×10^26 kg")
B = 33.04 #[T]
u_B = 0.0005 #[T]
μ_N = 5.0507837461 * 1e-27 #[J/T]
u_μ_N = 0.0000000015 * 1e-27 #[J/T]
μ_g = 0.0904 * μ_N #[J/T]
u_μ_g = np.sqrt((u_μ_N / μ_N)**2) * μ_g #[J/T]

I_g = 1/2
I_e = 3 / 2
m_g = [-1/2, 1/2]
m_e = [-3/2, -1/2, 1/2, 3/2]

def recoil_energy():
    # recoil energy of a single Fe atom
    E_R = E𝛾**2 / (2 * Fe_A * c**2)
    print(f"\nE𝛾**2= {E𝛾**2}keV^2, {Fe_A =}kg, {c**2 =}(m/s)^2")
    print(f"\nE_R : {E_R / meV} meV")
    return E_R

def read_files():
    # read data files and return numpy arrays
    data_KFe = np.array([int(value) for value in open('KFe.txt')])
    data_αFe = np.array([int(value) for value in open('alpha_Fe.txt')])
    x = np.arange(len(data_KFe))
    return x, data_KFe, data_αFe

def plot_data(E, data_file, fit, C_KFCN, C_αFe, αFe=False):
    plt.plot(E, data_file, linestyle='None', marker='.', color='plum',label=r"detected $\gamma-$rays")
    plt.plot(E, fit, color='teal',label=r"fit")

    plt.axvline(x=C_0,color='goldenrod', alpha=0.5, label=r"$C_0$")
    if αFe:
        plt.axvline(x=C_αFe,color='coral', alpha=0.5, label=r"$C_{\alpha Fe}$")
    else:    
        plt.axvline(x=C_KFCN,color='coral', alpha=0.5, label=r"$C_{KFCN}$")

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
    # print(f"\n{pars=}\n{perr=}")
    # return fit parameters
    return pars, perr, fit

def αFe_spectrum_fitting(E, data_αFe):
    # α-Iron data
    # initial parameters from eyeballing plot
    initial_guess = [560 * 1e3, 60 * 1e3, 90, 20, 50 * 1e3, 160, 20, 30 * 1e3, 225, 20, 30 * 1e3, 280, 20, 50 * 1e3, 345, 20, 60 * 1e3, 410, 20]
    pars_6, pcov = scipy.optimize.curve_fit(not_6_Lorentzians, E, data_αFe, p0=initial_guess)
    perr_6 = np.sqrt(np.diag(pcov))
    fit_6 = not_6_Lorentzians(E, *pars_6)
    # print(f"\n{pars_6=}\n{perr_6=}")
    # return fit parameters
    return pars_6, perr_6, fit_6

def energy_differences(pars_6, perr_6):
    # extracting values of peak locations
    E_0i =  np.sort(pars_6)[6:12]
    u_E_0i =  np.sort(perr_6)[:6]
    # print(f"\n{E_0i = }")
    LAMBDA_1 = E_0i[-2] - E_0i[2]
    u_LAMBDA_1 = np.sqrt(0.03**2 + 0.05**2)
    LAMBDA_2 = E_0i[-3] - E_0i[1]
    u_LAMBDA_2 = np.sqrt(0.03**2 + 0.05**2)

    lambda_1 = E_0i[5] - E_0i[4]
    u_lambda_1 = np.sqrt(0.03**2 + 0.03**2)
    lambda_2 = E_0i[4] - E_0i[3]
    u_lambda_2 = np.sqrt(0.03**2 + 0.05**2)
    lambda_3 = E_0i[2] - E_0i[1]
    u_lambda_3 = np.sqrt(0.03**2 + 0.05**2)
    lambda_4 = E_0i[1] - E_0i[0]
    u_lambda_4 = np.sqrt(0.03**2 + 0.03**2)

    mean_lambda = np.mean([lambda_1, lambda_2, lambda_3, lambda_4])
    u_mean_lambda = (1 / 4) * np.sqrt(0.04**2 + 0.06**2 + 0.04**2 + 0.06**2)

    return LAMBDA_1, u_LAMBDA_1, mean_lambda, u_mean_lambda, E_0i, u_E_0i

def Q8(LAMBDA_1, u_LAMBDA_1, mean_lambda, u_mean_lambda):
    # g value for ground state
    g_g = μ_g / (μ_N * I_g)
    u_g_g = np.sqrt((u_μ_g / μ_g)**2 + (u_μ_N / μ_N)**2) * g_g
    # Energy levels ground state
    E_g = []
    u_E_g = []
    for i in m_g:
        E_i = g_g * μ_N * B * i
        E_g.append(E_i)
        u_E_i = np.sqrt((u_g_g / g_g)**2 + (u_μ_N / μ_N)**2 +(u_B / B)**2) * E_i
        u_E_g.append(u_E_i)
    LAMBDA_g = g_g * μ_N * B
    u_LAMBDA_g = np.sqrt((u_g_g / g_g)**2 + (u_μ_N / μ_N)**2 + (u_B / B)**2) * LAMBDA_g

    v = (LAMBDA_g * c) / E𝛾
    u_v = np.sqrt((u_LAMBDA_g / LAMBDA_g)**2 + (u_E𝛾 / E𝛾)**2) * v
    print(f"\n v = ({v * 1e3:.2f} ± {u_v * 1e3:.2f}) mm/s")

    K_18 = v / LAMBDA_1
    u_K_18 =  np.sqrt((u_v / v)**2 + (u_LAMBDA_1 / LAMBDA_1)**2) * K_18
    print(f"\nK_18 = ({K_18 * 1e3:.4f} ± {u_K_18 * 1e3:.4f}) mm/s/channel")

    v_mean_lambda = mean_lambda * K_18
    u_v_mean_lambda = np.sqrt((u_mean_lambda / mean_lambda)**2 + (u_K_18 / K_18)**2) * v_mean_lambda
    print(f"\nv_mean_lambda = ({v_mean_lambda * 1e3:.3f} ± {u_v_mean_lambda * 1e3:.3f}) mm/s")

    Delta_E = ((E𝛾 / keV)* v_mean_lambda) / c
    u_Delta_E = np.sqrt((u_E𝛾 / E𝛾)**2 + (u_v_mean_lambda/ v_mean_lambda)**2) * Delta_E
    print(f"\nDelta_E = ({Delta_E * keV * 1e26:.3f} ± {u_Delta_E * keV * 1e26:.3f}) * 1e-26 J")
    
    g_e = Delta_E * keV / (μ_N * B)
    u_g_e = np.sqrt((u_Delta_E / Delta_E)**2 + (u_μ_N / μ_N)**2 + + (u_B / B)**2) * g_e
    print(f"\ng_e = ({g_e:.4f} ± {u_g_e:.4f}) ")

    μ_e = g_e * μ_N * (3 / 2)
    u_μ_e = np.sqrt((u_g_e/ g_e)**2 + (u_μ_N / μ_N)**2) * μ_e
    print(f"\nμ_e = ({μ_e * 1e28:.2f} ± {u_μ_e * 1e28:.2f}) J/T")
    print(f"\nμ_e = ({μ_e / μ_N:.4f} ± {u_μ_e / μ_N:.4f}) μ_N")
    return K_18, u_K_18


def Q9(pars, perr, K_18, u_K18):
    v_αFe = 711 * 1e-3 #[V]
    u_v_αFe = 2 * 1e-3 #[V]
    v_KFCN = 160 * 1e-3 #[V]
    u_v_KFCN = 1 * 1e-3 #[V]

    K_04 = v_αFe * K_18 / v_KFCN
    u_K_04 = np.sqrt((u_v_αFe / v_αFe)**2 + (u_K_18 / K_18)**2 +(u_v_KFCN / v_KFCN)**2) * K_04

    print(f"\nK_04 = ({K_04 * 1e4:.2f} ± {u_K_04 * 1e4:.2f}) * 1e-4 mm/s/channel")
    GAMMA_v = K_04 * pars[3]
    u_GAMMA_v = np.sqrt((u_K_04 / K_04)**2 +(perr[3] / pars[3])**2) * GAMMA_v
    print(f"\nGAMMA_v = ({GAMMA_v * 1e3:.2f} ± {u_GAMMA_v * 1e3:.2f}) * 1e-3 mm/s")
    v_Q_1 = (7.479 * 1e-28 * c) / E𝛾
    print(f"\nv_Q_1 = {v_Q_1 * 1e3:.1f} mm/s")

    convolv_v_Q1 = v_Q_1 * 2.3
    print(f"\nconvolv_v_Q1 = {convolv_v_Q1 * 1e3:.1f} mm/s")

    return K_04, u_K_04

def Q10(pars,perr, pars_6, perr_6, K_04, u_K_04, K_18, u_K_18, E_0i, u_E_0i):
    C_KFCN = pars[2]
    u_C_KFCN = perr[2]
    print(f"\nC_KFCN = ({C_KFCN:.3f} ± {u_C_KFCN:.3f})")

    C_KFCN_C0 = C_KFCN - C_0
    u_C_KFCN_C0 = np.sqrt(u_C_KFCN**2 + u_C_0**2)
    print(f"\n{C_KFCN_C0 = :.1f} ± {u_C_KFCN_C0 = :.1f}")
    
    IS_KFCN = C_KFCN_C0 * K_04
    u_IS_KFCN = np.sqrt((u_C_KFCN_C0/ C_KFCN_C0)**2 + (u_K_04 / K_04)**2) * IS_KFCN
    print(f"\nIS_KFCN = ({IS_KFCN * 1e3:.2f} ± {u_IS_KFCN * 1e3:.2f}) mm/s")


    C_αFe = np.mean(E_0i)
    u_C_αFe = (1/6) * np.sqrt(sum(x**2 for x in u_E_0i))
    print(f"\nC_αFe = {C_αFe:.3f} ± {u_C_αFe:.3f}")

    C_αFe_C0 = C_αFe - C_0
    u_C_αFe_C0 = np.sqrt(u_C_αFe**2 + u_C_0**2)
    print(f"\nC_αFe_C0 = {C_αFe_C0:.2f} ± {u_C_αFe_C0:.2f}")

    IS_αFe = C_αFe_C0 * K_18

    u_IS_αFe = np.sqrt((u_C_αFe_C0 / C_αFe_C0)**2 + (u_K_18 / K_18)**2) * IS_αFe
    print(f"\nIS_αFe = ({IS_αFe * 1e3:.4f} ± {u_IS_αFe * 1e3:.4f}) mm/s")

    diff = IS_KFCN - IS_αFe
    u_diff = np.sqrt(u_IS_KFCN**2 + u_IS_αFe**2)
    print(f"\ndiff = ({diff * 1e3:.2f} ± {u_diff * 1e3:.2f}) mm/s")

    return C_KFCN, u_C_KFCN, C_αFe, u_C_αFe






#*~ function calls ~*#
x, data_KFe, data_αFe = read_files()
pars, perr, fit = KFe_spectrum_fitting(E, data_KFe)

pars_6, perr_6, fit_6 = αFe_spectrum_fitting(E, data_αFe)

LAMBDA_1, u_LAMBDA_1, mean_lambda, u_mean_lambda, E_0i, u_E_0i = energy_differences(pars_6, perr_6)

K_18, u_K_18 = Q8(LAMBDA_1, u_LAMBDA_1, mean_lambda, u_mean_lambda)

K_04, u_K_04 = Q9(pars, perr, K_18, u_K_18)

C_KFCN, u_C_KFCN, C_αFe, u_C_αFe = Q10(pars,perr, pars_6, perr_6, K_04, u_K_04, K_18, u_K_18, E_0i, u_E_0i)


plot_data(E, data_KFe, fit, C_KFCN, C_αFe, False)
plot_data(E, data_αFe, fit_6, C_KFCN, C_αFe, True)
