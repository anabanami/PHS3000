# PHS3000 - Uncertainties and advanced fitting -Part B
# Analysing nuclear decay data
# Ana Fabela, 10/08/2020

import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS3000 as spa
plt.rcParams['figure.dpi'] = 150

data = spa.tutorials.fitting.part_b_data
# print(data)

time = data[:, 0] # first column (time elapsed in seconds [s])
A = data[:, 1] # second column (A is activity [dimensionless])
u_A = data[:, 2] # third column (uncertainty in activity [dimensionless])

# Non-linear fit
nonlinear = spa.make_lmfit_model("A_0 * np.exp(-lambda_ * x) ")
# half-life of silver is ~41days ~~3e6 seconds
# ln(2) / 3e6 ~~ 2e-7 [s**-1]
# then, our itinial guesses are:
nonlinear_params = nonlinear.make_params(A_0=20,lambda_=2e-7)
fit_results = spa.model_fit(nonlinear, nonlinear_params, time, A, u_y=u_A)
# extracting our parameters
fit_parameters = spa.get_fit_parameters(fit_results)
print(fit_parameters)
# making our non-linear fit with one sigma uncertainty
A_fit = fit_results.best_fit
u_A_fit = fit_results.eval_uncertainty(sigma=1)


# # creating figure environment
plt.figure(1)
# # plot
plt.errorbar(
            time, A, xerr=0, yerr=u_A,
            marker="o", linestyle="None",
            label="Activity data"
)

plt.plot(
        time, A_fit, marker="None",
        linestyle="-", 
        label="non-linear fit"
)
plt.fill_between(
                time, A_fit - u_A_fit,
                A_fit + u_A_fit,
                alpha=0.5,
                label="uncertainty in non-linear fit"
)
plt.title("Figure 1: Non-linear fit of silver decay data")
plt.xlabel("t [s]")
plt.ylabel("A [arb]")
plt.legend()
spa.savefig('nonlinear_silver_decay_data.png')
# # showmewhatyougot
plt.show()

########################################################

# Linear fit
time_lin = time # first column (time elapsed in seconds [s])
A_lin = np.log(A) # second column A is linearised by ln(A) [dimensionless]
u_A_lin = u_A / A # uncertainty in log(A) is fractional uncertainty in activity [dimensionless]
linear_results = spa.linear_fit(time_lin, A_lin, u_y=u_A_lin)

# # making our linear fit with one sigma uncertainty
A_lin_fit = linear_results.best_fit
u_A_lin_fit = linear_results.eval_uncertainty(sigma=1)

plt.figure(2)
# plot
plt.errorbar(
            time_lin, A_lin, xerr=0, yerr=u_A_lin,
            marker="o", linestyle="None",
            label="Activity data"
)

plt.plot(
        time_lin, A_lin_fit, marker="None",
        linestyle="-", 
        label="linear fit"
)
plt.fill_between(
                time, A_lin_fit - u_A_lin_fit,
                A_lin_fit + u_A_lin_fit,
                alpha=0.5,
                label="uncertainty in linear fit"
)
plt.title("Figure 2: Linear fit of silver decay data")
plt.xlabel("t [s]")
plt.ylabel("ln(A) [arb]")
plt.legend()
spa.savefig('linear_silver_decay_data.png')
plt.show()

########################################################

# Investigating the two fits
# Plot the residuals of each fit in a new figure, 
# using the uncertainty of your raw/linearised data as the error bars.

# Non-linear fit residuals
nonlinear_residuals = A_fit - A # non-linear residuals (non-linear best fit - raw data)

plt.figure(3)
# plot
plt.errorbar(
            time, nonlinear_residuals, xerr=0, yerr=u_A,
            marker="o", linestyle="None",
            label="Residuals (non-linear fit-data)"
)
plt.plot([time[0], time[-1]], [0,0], color="k")
plt.title("Figure 3: Residuals of non-linear fit of silver decay data")
plt.xlabel("t [s]")
plt.ylabel("Residuals [arb]")
plt.legend()
spa.savefig('non-linear_residuals_silver_decay_data.png')
plt.show()
########################################################
# Linear fit residuals
linear_residuals = A_lin_fit - A_lin # linear residuals (linear best fit - linearised data)

plt.figure(4)
# plot
plt.errorbar(
            time_lin, linear_residuals, xerr=0, yerr=u_A_lin,
            marker="o", linestyle="None",
            label="Residuals (linear fit-data)"
)
plt.plot([time[0], time[-1]], [0,0], color="k")
plt.title("Figure 4: Residuals of linear fit of silver decay data")
plt.xlabel("t [s]")
plt.ylabel("ln(A) [arb]")
plt.legend()
spa.savefig('linear_residuals_silver_decay_data.png')
plt.show()

 # data at early times when the activity is high
#produces more meaningful results since fractional uncertainty is smaller

