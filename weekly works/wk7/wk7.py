# PHS3000
# Week 7 - UNCERTAINTY -
# Ana Fabela Hinojosa, 16/09/2020
import monashspa.PHS3000 as spa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import special
plt.rcParams['figure.dpi'] = 200

# Exercise 1. Uncertainty calculation for a function.
# Calculate the uncertainty in the measurement
# of the wavelength 𝜆 assuming that all the uncertainty 
# in the measurement is due to the uncertainty in 
# the measuring angle 𝜃.
# The slit width is 2 𝜇m.

𝜇m = 1e-6 # [m]

d = 2 * 𝜇m # [m]
𝜃 = 20 * (np.pi / 180) # [rad]
u_𝜃 = 3 * (np.pi / 180) # [rad]

𝜆 = d * np.sin(𝜃) # [m]
u_𝜆 = d * np.cos(𝜃) * u_𝜃 # [m]

print(f"\n(Ex.1) 𝜆 =({𝜆 * 1e8:.1f} ± {u_𝜆 * 1e8:.1f}) ×10^-8 m")

# Exercise 2. Uncertainty calculations using the 
# general formula for error propagation.

#The parameters in 𝑞 are:
x = 10
u_x = 2
y = 7
u_y = 1
𝜃 = 40 * (np.pi / 180) # [rad]
u_𝜃 = 3 * (np.pi / 180) # [rad]

# Calculate the value of 𝑞 at these conditions and its uncertainty.

𝑞 = (x + 2) / (x + y * np.cos(4 * 𝜃))
# We calculate partial derivatives wrt all parameters in 𝑞
# (we did this analytically in paper and simply wrote the exressions here)
𝑞_x = ((x + y * np.cos(4 * 𝜃)) - (x + 2)) / (x + y * np.cos(4 * 𝜃))**2
𝑞_y = -(x + 2) * np.cos(4 * 𝜃) / (x + y * np.cos(4 * 𝜃))**2
𝑞_𝜃 = (x + 2) * 4 * y * np.sin(4 * 𝜃) / (x + y * np.cos(4 * 𝜃))**2

# By the general formula for error propagation
u_𝑞 = np.sqrt((𝑞_x * u_x)**2 + (𝑞_y * u_y)**2 + (𝑞_𝜃 * u_𝜃)**2)

print(f"\n(Ex.2) 𝑞 = {𝑞:.1f} ± {u_𝑞:.1f}")

# Exercise 3. Uncertainty calculations when there is no analytic expression. 
# ie. You do not know the mathematical form of the function

mm = 1e-3 # [m]

r = 0.7 * mm # [m]
u_r = 0.05 * mm # [m]
d = 3.1 * mm # [m]
u_d = 0.1 * mm # [m]
I = 1.25 * 1e3 # [W/m^2]
u_I = 0.05 *1e3 # [W/m^2]

temperature_rise = spa.tutorials.model_1(r,d,I)

# Recipe:
# 1.Calculate the value of your function q(x,y,z).
# 2.Calculate q(x + Δx,y,z).
# 3.Calculate q(x + Δx,y,z)-q(x,y,z). This gives Δq(x).
# 4.Repeat steps 2-3 for all parameters.
# 5.Calculate the final uncertainty.
# (by adding all the individual absolute uncertainties in quadrature.) 

u_q_r = temperature_rise - spa.tutorials.model_1(r + u_r,d,I)
u_q_d = temperature_rise - spa.tutorials.model_1(r,d + u_d,I)
u_q_I = temperature_rise - spa.tutorials.model_1(r,d,I + u_I)

u_q = np.sqrt((u_q_r)**2 + (u_q_d)**2 + (u_q_I)**2)

# What is the expected temperature rise?
print(f"\n(Ex.3) The expected temperature rise = {temperature_rise:.2f} ± {u_q:.2f}")

# Exercise 4.  Exercise on the rejection of data
# Chauvenet criterion: 
# If the expected number of measurements as deviated as the suspect one 
# is less the 0.5, then reject the suspect measurement.

def stats(tracks):
    𝜇 = np.mean(tracks)
    𝜎 = np.std(tracks)
    return 𝜇, 𝜎

# step 1
def deviants(tracks, 𝜇, 𝜎):
    for measurement in tracks:
        suspect = 𝜇 - measurement
        t = abs(suspect / 𝜎)
        if t > 2.0:
            print(f"\n{measurement} is {t:.2f}𝜎 away from {𝜇 = :.2f}")
            return measurement, t

# step 2
def probability(t):
    # option 2: Using the error function.
    return special.erf(t / np.sqrt(2))

# step 3
def N_measurements(P, tracks):
    return (1 - P) * len(tracks)

def Chauvenet(tracks):
    # step 0
    𝜇, 𝜎 = stats(tracks)
    # step 1
    result = deviants(tracks, 𝜇, 𝜎)
    if result is None:
        return True
    weirdo, t = result
    # step 2
    P = probability(t)
    # step 3
    N = N_measurements(P, tracks)
    print("The expected number of deviant measurements is")
    # step 4
    if N < 0.5:
        print(f"{N = :.2f}, therefore rejection can be considered.")
        tracks.remove(weirdo)
        print("\nAfter applying the criterion, our list is")
        print(f"{tracks}.")
    return False

print(f"\n(Ex.4) Chauvenet's criterion")
# RECIPE
# step 1
# Identify the most deviant data point and apply the test
# step 2
# Calculate the probability of a measurement occurring 
# this far away from 𝜇.
# step 3
# Calculate the number of measurements in the experiment that would be 
# expected to be as deviated as the suspect result
# step 4
# The expected number of deviant measurements is 
tracks = [11, 9, 13, 16, 8, 10, 5, 11, 9, 12, 12, 13, 10, 14]
print(f"{tracks = }")
# Chauvenet(tracks)
while True:
    done = Chauvenet(tracks)
    if done:
        break

print(f"\nOur list: {tracks = }")

