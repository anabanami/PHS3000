# PHS3000 -Uncertainties and advanced fitting -Part C1
# Building composite models
# Ana Fabela, 10/08/2020

import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS3000 as spa
plt.rcParams['figure.dpi'] = 150

data = spa.tutorials.fitting.part_c1_data
# slicing data (data is cake)
x = data[:, 0]
y = data[:, 1]
# print(f"{data=}")

# Python task: Build a composite model 
# consisting of the sum of three components: two Gaussians and a constant offset term.
gauss1 = spa.make_lmfit_model("A1 * np.exp(-(x - B1)**2 / D1**2)", name="Gaussian 1")
gauss2 = spa.make_lmfit_model("A2 * np.exp(-(x - B2)**2 / D2**2)", name="Gaussian 2")
offset = spa.make_lmfit_model("c + 0 * x", name="Offset")

model = gauss1 + gauss2 + offset
# Python task: Perform a fit to your data using this composite model, 
# and plot the results (including fit uncertainty. In order to fit successfully,
# you will need to provide initial guesses for the 7 free parameters of your fit.

# initial guesses for parameters based on data
params = model.make_params(A1=19.5, A2=18.5, B1=96, B2=164, c=2.5)
params.add('D2',expr='D1') # new line
params.add('FWHM', expr='D1')
params.add('FWHM', expr='2 * sqrt(2 * log(2)) * D1')

fit_results = spa.model_fit(model, params, x, y)
# extracting our parameters
fit_parameters = spa.get_fit_parameters(fit_results)
print(fit_parameters)
# making our fit with one sigma uncertainty
y_fit = fit_results.best_fit
u_y_fit = fit_results.eval_uncertainty(sigma=1)

plt.figure(5)
# data
plt.plot(
        x, y, marker="o",
        linestyle="None", 
        label="raw data"
)
# fit line
plt.plot(
        x, y_fit, marker="None",
        linestyle="-", 
        label="non-linear fit"
)
# uncertainty
plt.fill_between(
                x, y_fit - u_y_fit,
                y_fit + u_y_fit,
                alpha=0.5,
                label="uncertainty in non-linear fit"
)
plt.title("Figure 5: Plot of raw data with fit (including components)")
plt.xlabel("x")
plt.ylabel("y")
for component_name, component_y_values in fit_results.eval_components().items():
    plt.plot(x,component_y_values,label=component_name)

plt.legend(loc='upper left',prop={"size":7})
spa.savefig('gaussian_model.png')
plt.show()