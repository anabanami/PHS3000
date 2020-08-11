# PHS3000 -Uncertainties and advanced fitting -Part C2
# Analysing silver decay data
# Ana Fabela, 11/08/2020

import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS3000 as spa
from pprint import pprint
plt.rcParams['figure.dpi'] = 150

data = spa.tutorials.fitting.part_c2_data

# data columns
t = data[:,0]
A = data[:,1]
# print(f"{A=}")
u_A = data[:,2]
# print(f"{u_A=}")

# setting up the composite model
ag110_model = spa.make_lmfit_model("AG110_A_0 * exp(-AG110_lambda * x)", name=r"$Ag_{110}$")
ag108_model = spa.make_lmfit_model("AG108_A_0 * exp(-AG108_lambda * x)", name=r"$Ag_{108}$")
offset = spa.make_lmfit_model("c + 0 * x", name="Offset")
model = ag110_model + ag108_model + offset

# initial guesses for parameters based on data and wikipedia
# https://en.wikipedia.org/wiki/Isotopes_of_silver
params = model.make_params(
                            AG110_A_0=50, AG108_A_0=50,
                            AG110_lambda=0.03, AG108_lambda=0.005,
                            c=0.5
)
fit_results = spa.model_fit(model, params, t, A, u_y=u_A)

# extracting our parameters
fit_parameters = spa.get_fit_parameters(fit_results)
pprint(fit_parameters)

# making our fit with one sigma uncertainty
A_fit = fit_results.best_fit
u_A_fit = fit_results.eval_uncertainty(sigma=1)


plt.figure(6)
plt.errorbar(
            t, A, xerr=0, yerr=u_A,
            marker="o", linestyle="None",
            label="Activity data"
)
# fit line
plt.plot(
        t, A_fit, marker="None",
        linestyle="-", 
        label="non-linear fit"
)
# uncertainty
plt.fill_between(
                t, A_fit - u_A_fit,
                A_fit + u_A_fit,
                alpha=0.5,
                label="uncertainty in non-linear fit"
)

plt.title("Figure 6: Non-linear plot of dual silver isotope decay data")
plt.xlabel("t [s]")
plt.ylabel("Activity [counts/s]")
for component_name, component_y_values in fit_results.eval_components().items():
    plt.plot(t,component_y_values,label=component_name)
plt.legend(loc='upper right',prop={"size":7})
spa.savefig('nonlinear_dual_silver_decay.png')
plt.show()

# Non-linear fit residuals
nonlinear_residuals = A_fit - A # non-linear residuals (non-linear best fit - raw data)

plt.figure(7)
plt.errorbar(
            t, nonlinear_residuals, xerr=0, yerr=u_A,
            marker="o", linestyle="None",
            label="Residuals (non-linear fit-data)"
)
plt.plot([t[0], t[-1]], [0,0], color="k")
plt.title("Figure 7: Residuals of non-linear dual silver isotope decay data")
plt.xlabel("t [s]")
plt.ylabel("Residuals [arb]")
plt.legend()
spa.savefig('non-linear_residuals_dual_silver_decay_data.png')
plt.show()

# we can see that we have residuals that are MOSTLY evenly spaced 
# around 0. 

# How do our results compare with the expected values for half-life 
# (Ag108: 142.2±0.6 seconds, Ag110: 24.6±0.2 seconds)? 

# Our dictionary returns the calculated decay constant for each isotope,
# which we can use to calculate the half-life:
AG108_halflife = np.log(2) / fit_parameters['AG108_lambda']
AG110_halflife = np.log(2) / fit_parameters['AG110_lambda']

# propagating uncertainties using our dictionary values
u_AG108_halflife = fit_parameters['u_AG108_lambda'] / fit_parameters['AG108_lambda'] * AG108_halflife
u_AG110_halflife = fit_parameters['u_AG110_lambda'] / fit_parameters['AG110_lambda'] * AG110_halflife

print(f"\nCalculated half-life of Ag_108: \n{AG108_halflife:.1f} ± {u_AG108_halflife:.1f} seconds")
print(f"Calculated half-life of Ag_110:\n{AG110_halflife:.1f} ± {u_AG110_halflife:.1f} seconds")

print("Our result for the half-life of Ag_108 agrees with the theoretical value." )
print("Whilst our result for the half-life of Ag_110 is 2 standard deviations\naway from the theoretical value.")

# What is your result for the background level? Does it make physical sense?
background = fit_parameters['c']
u_background = fit_parameters['u_c']

print(f"\nCalculated background level decay:{background:.2f} ± {u_background:.2f} counts")
print(f"\nOur calculated counts appear to be very close to zero, obviously this value cannot be negative.\nWe only consider the positive range of this obtained value")