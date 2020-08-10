# PHS3000 -Introduction to Python -part B
# Using a pendulum to measure the gravitational constant
# Ana Fabela, 03/08/2020

import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS3000 as spa
plt.rcParams['figure.dpi'] = 150

data = [
        #length[m], u(length)[m], period1[s],period2[s],period3[s]
        [0.300, 0.003, 1.2, 1.0, 1.1],
        [0.600, 0.003, 1.4, 1.5, 1.7], 
        [0.900, 0.003, 2.0, 2.1, 1.9],
        [1.200, 0.003, 2.4, 2.1, 2.3],
        [1.500, 0.003, 2.3, 2.6, 2.5],
        [1.800, 0.003, 2.6, 2.5, 2.8]
]

data = np.array(data)

length = data[:, 0]
u_length = data[:, 1]

## The last three columns of data are repeated measurements 
## of the period. We want to use these repeated measurements
## to calculate the average period and the corresponding 
## uncertainty in the average period. 
## To do this, you need to extract the last three columns
## of data into a separate two-dimensional array, 
## which we'll chose to name as "period_data":
period_data = data[:, 2:5]
## To average across the three columns: axis=1
## to average down each column: axis=0
## to average in both directions simultaneously: 
## np.mean(period_data)
avg_period = np.mean(period_data,axis=1)

## the average period was calculated from repeated measurements 
## of the period, the uncertainty in the average period is 
## the standard deviation of the sample
## divided by the square root of the number of repeated measurements. 
u_avg_period = np.std(period_data,axis=1, ddof=1) / np.sqrt(3)
## "ddof=1"; the name of this argument stands for 
## "Delta Degrees of Freedom", 
## which is a term used in Statistics. For our purposes,
## setting "ddof=1" means that we're calculating 
## the standard deviation of the sample,
## instead of the standard deviation of the population

## creating new figure environment
# plt.figure(1)
# # plot
# plt.title("Figure 1: Non-linear plot of pendulum data")
# plt.errorbar(
#             length, avg_period, xerr=u_length,yerr=u_avg_period,
#              marker="o", linestyle="None"
# )
# plt.xlabel("L [m]")
# plt.ylabel(r"$T_{avg}$ [s]")
# plt.xlim([0, 2])
# plt.ylim([0, 3])
# spa.savefig('figure1.png')

## showmewhatchugot
# plt.show()
####################################################################
## Linearised data
avg_period_sq = avg_period**2
u_avg_period_sq = 2 * avg_period * u_avg_period

## LINE OF BEST FIT
fit_results = spa.linear_fit(
                            length, avg_period_sq, 
                            u_y=u_avg_period_sq
)
## best_fit is attribute of fit_results
## It is an array of y-values corresponding to the line of best fit
y_fit = fit_results.best_fit
## "eval_uncertainty()" is a method of the object "fit_results". 
## returns an array of the uncertainty in the fitted y-values,
## using the fitted y-values obtained from best_fit,
## the value of the uncertainty will correspond to one sigma.
u_y_fit = fit_results.eval_uncertainty(sigma=1)

plt.figure(2)
plt.title("Figure 2: Linear plot of pendulum data")
plt.errorbar(
            length,avg_period_sq, xerr=u_length,
            yerr=u_avg_period_sq,
            marker="o", linestyle="None", color="black",
            label="pendulum data"
)
plt.plot(
        length, y_fit, marker="None",
        linestyle="-", color="black",
        label="linear fit"
)
plt.fill_between(
                length,y_fit - u_y_fit,
                y_fit + u_y_fit, 
                color="lightgrey",
                label="uncertainty in linear fit"
)
plt.xlabel("L [m]")
plt.ylabel(r"$T^{2}_{avg} [s^2]$")
plt.xlim([0, 2])
plt.ylim([0, 8])
plt.legend()#bbox_to_anchor=(1,1))
spa.savefig('figure2.png')
# plt.show()
####################################################################

## Calculating g from our data
## This function returns a dictionary containing the intercept, 
## uncertainty in the intercept, gradient,
## and uncertainty in the gradient, from the linear line of best fit.
fit_parameters = spa.get_fit_parameters(fit_results)
print(fit_parameters)

## in this case the y-intercept corresponds
## to the square of the period, T**2 in units of s**2)

gradient = fit_parameters["slope"]
u_gradient = fit_parameters["u_slope"]
## We have extracted the value corresponding to the key "slope" 
## from the dictionary we named "fit_parameters", and chosen to store
## this value in the variable name "gradient". Then we've done 
## the same thing for the uncertainty in the gradient.



measured_g = 4* np.pi**2 / gradient
u_measured_g = 4* np.pi**2 * u_gradient / (gradient**2)
print(f"measured gravitational acceleration: g = {measured_g} +/- {u_measured_g} m/s^2")