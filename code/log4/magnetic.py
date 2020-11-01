# PHS3000
# Magnetic susceptibility
# Ana Fabela, 29/10/2020
import os
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from physunits import *


plt.rcParams['figure.dpi'] = 150
read_folder = Path('data')

# Globals
μ0 = 1.25663706212e-6 * H / m # vacuum permeability
k = 1.38064852e-23 * m**2 * kg /(s**2 * K) # Boltzmann constant
T = 294 * K
u_T = 1 * K
# from eyeballing calibration data
initial_linear_guess = [0.3, 0]
# from eyeballing samples data
# initial_guess = []

def load_data(filename):
    xs = []
    ys = []
    for line in filename:
        x, y = line.split(',')
        xs.append(float(x))
        ys.append(float(y))
    return np.array(xs), np.array(ys)

def line_fit(x, m, c):
    # print(f"{m=},{x}")
    return m * x + c

# def other_kind_fit():
#     return 

def fitting_calibration_data(xs, ys, initial_linear_guess):
    pars, pcov = scipy.optimize.curve_fit(line_fit, xs, ys, p0=initial_linear_guess)
    perr = np.sqrt(np.diag(pcov))
    print(f"{pars}, {perr}\n")
    # print(f"{xs =}")
    linear_fit = line_fit(xs, *pars)
    return pars, perr, linear_fit

# def fitting_magnetization_data(xs, ys, initial_guess):
#     pars, pcov = scipy.optimize.curve_fit(other_kind_fit, xs, ys, p0=initial_guess)
#     perr = np.sqrt(np.diag(pcov))
#     print(f"{pars}, {perr}")
#     # print(f"{xs =}")
#     fit = other_kind_fit(xs, *pars)
#     return pars, perr, fit

def plot_data(xs, ys, u_xs, u_ys, fit, filename, calibration=False):
    if calibration:
        plt.plot(xs, fit, color='teal',label=r"fit")
        plt.errorbar(xs, ys,
            xerr=u_xs,yerr=u_ys,color='orange', 
            marker='None',linestyle='None', label="Magnetic field"
        )
        plt.ylabel('B / mT')
    else:
        plt.errorbar(xs, ys,
            xerr=u_xs, yerr=u_ys, color='olive', 
            marker='None',linestyle='None', label="Mass"
        )
        plt.ylabel('M / g')

    plt.xlabel("Current / A")
    plt.title(f"{filename}")
    plt.legend()
    plt.show()

def propagate_uncertainty(xs, ys, calibration=False):
    u_xs = []
    u_ys = []
    print("New sample")
    for x in xs:
        u_x = 0.012 * x
        u_xs.append(u_x)
        print(f"{u_x = }")
    if calibration:
        # uncertainty in Magnetic field measurements: using typical uncertainty
        for y in ys:
            u_y = 0.050 * y + 0.020 * 300 # [mT]
            # print(f"{u_x = } , {u_y = }")
            u_ys.append(u_y)
    else:
        # uncertainty in mass measurements: using repeatability and linearity 
        for y in ys:
            u_y = np.sqrt(0.0001**2 + (0.0002/120 * y)**2 + 0.00005**2) # [g]
            print(f"{u_y = }")
            u_ys.append(u_y)
    return np.array(u_xs), np.array(u_ys)


def nickel_sample(xs, ys, u_xs, u_ys):
    # determine S note: J=S
    g = 2
    A = 38.5e-6 * m**2





def main():
    files = list(os.listdir(path=read_folder))
    files.sort(key=lambda name: int(name.strip('.csv').split('_')[-1]) if name[-5].isdigit() else -1)
    # print(files)
    file_names = []

    for i, file in enumerate(files):
        name = file.split(".")[0]
        file_names.append(name)
        # print(name)
        # print(i, file)
        file = open(read_folder / file)
        xs, ys = load_data(file)
        if i == 0:
            pars, perr, fit = fitting_calibration_data(xs, ys, initial_linear_guess)
            u_xs, u_ys = propagate_uncertainty(xs, ys, calibration=True)
            plot_data(xs, ys, u_xs, u_ys, fit, name, calibration=True)

        elif i == 1:
            u_xs, u_ys = propagate_uncertainty(xs, ys, calibration=False)
            plot_data(xs, ys,u_xs, u_ys, fit, name, calibration=False)
            dummy_xs = xs
            dummy_ys = ys
            u_dummy_xs = u_xs
            u_dummy_ys = u_ys

        elif i > 1:
            u_xs, u_ys = propagate_uncertainty(xs, ys, calibration=False)
            xs, ys = xs - dummy_xs, ys - dummy_ys
            u_xs, u_ys = np.sqrt(u_xs**2 + u_dummy_xs**2), np.sqrt(u_ys**2 + u_dummy_ys**2)
            plot_data(xs, ys,u_xs, u_ys, fit, name, calibration=False)



        

main()

