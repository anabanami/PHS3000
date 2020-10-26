# PHS3000
# Week 9 - CHI SQUARED -
# Ana Fabela Hinojosa, 16/09/2020
import monashspa.PHS3000 as spa
import pandas
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import chi2
from scipy.optimize import curve_fit
from scipy.special import factorial


params = {'legend.fontsize': 5,
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rcParams['figure.dpi'] = 200

# Exercise 1. Calculating chi-squared distribution metrics. 

degrees_freedom = [3, 5, 7, 10, 20, 50, 100, 1000]

def reduced_Chi2(degrees_freedom):
    # reduced chi2 and probability of reduced chi2 
    p_arrays = []
    red_chi2_values = []
    p_max_values = []
    for 𝜈 in degrees_freedom:
        red_chi2 = np.linspace(0, 5000, 1e6) / 𝜈
        p_red_chi2 = chi2.pdf(red_chi2 * 𝜈, 𝜈) * 𝜈

        argmax = np.argmax(p_red_chi2)
        p_max = p_red_chi2[argmax]
        probable_red_chi2 = red_chi2[argmax]

        p_arrays.append(p_red_chi2)
        p_max_values.append(p_max)
        red_chi2_values.append(probable_red_chi2)
    # print(f"{red_chi2_values = } , \n{p_max_values = }")
    # print(f"\n{red_chi2 = } , \n{p_arrays = }\n")

    return red_chi2, p_arrays, red_chi2_values, p_max_values

def Task1(degrees_freedom, red_chi2, p_arrays, red_chi2_values, p_max_values):
    # Determine the most probable value of the reduced chi-squared 
    # distribution (to 3 significant figures) for data sets with
    # degrees of freedom (𝜈)
    # plot reduced chi2 vs probability of reduced chi2
    for i, 𝜈 in enumerate(degrees_freedom):
        value = red_chi2_values[i]
        plt.plot(red_chi2, p_arrays[i], label=fR"$\nu$={𝜈}, $\chi^2_r$={value:.3f} ")
    plt.legend()
    plt.xlim(0, 1.25)
    plt.xlabel(R"$\chi^2_r$")
    plt.ylabel(R"$P(\chi^2,\nu)$")
    plt.title("Task 1")   
    plt.show()

def Task2(degrees_freedom, red_chi2, red_chi2_values):
    # Calculate the corresponding P-values for the 
    # peak chi-squared values found above
    P_values = []
    for i, 𝜈 in enumerate(degrees_freedom):
        P = 1 - chi2.cdf(red_chi2_values[i] * 𝜈, 𝜈)
        P_values.append(P)
        print(f"for {𝜈 = }, the P-value is {P:.3f}")
    return P_values

def Task3(degrees_freedom, red_chi2, p_arrays, red_chi2_values, p_max_values, P_values):
    # Plot both the most probable value of the reduced chi-squared distribution 
    # and the P-value as a function of the degrees of freedom for values up to 𝜈=100.  
    plt.plot(degrees_freedom[:-1], red_chi2_values[:-1])
    plt.ylabel(R"Peak $\chi^2_r$")
    plt.xlabel("degrees of freedom")
    plt.title("Task 3.1")
    plt.show()

    plt.plot(degrees_freedom[:-1], P_values[:-1])
    plt.ylabel("Peak P-values")
    plt.xlabel("degrees of freedom")
    plt.title("Task 3.2")   
    plt.show()
    # What do you conclude about the rule of 𝜒^2_r≈1 and 𝑃≈0.5  being indicative of a good fit, for small and large data sets?
    print("\nThe most probable value for chi2_r approaches 1 with more degrees of freedom,\n ie. Accuracy decreases with less degrees of freedom.\n") 

def read_files():
    polynomial_data = pandas.read_csv('polynomial_data.csv').to_numpy()
    radioactive_data = pandas.read_csv('radioactive_counts.csv', header=None).to_numpy()
    # print("\n", polynomial_data, type(polynomial_data))
    return polynomial_data, radioactive_data

# Exercise 2. Chi-squared testing and Occam’s razor.

def polynomial_coeff(data, deg):
    # determine coefficients for different degree polynomials
    x_array = np.asarray([row[0] for row in data])
    y_array = np.asarray([row[1] for row in data])
    yerr_array = np.asarray([row[2] for row in data])
    weights = [(1 / yerr)**2 for yerr in yerr_array]

    degN_coeff = poly.polyfit(x_array, y_array, deg, w=weights)

    return x_array, y_array, yerr_array, degN_coeff

def poly_fit(x_array, *degN_coeff):
    # polynomial fit function
    terms = []
    for i, coeff in enumerate(degN_coeff):
        term = coeff * x_array**i
        terms.append(term)
    return sum(terms)

def Exercise_2(polynomial_data, deg):
    # Fit polynomials of order 2 through to 8 to the data using a weighted fit, 
    # and by using chi-squared tests, determine which model is the most appropriate.
    chi_squared_values = []
    red_chi_squared_values = []
    for order in deg:
        # degrees of freedom for each polynomial
        𝜈 = len(polynomial_data) - order
        x_array, y_array, yerr_array, degN_coeff = polynomial_coeff(polynomial_data, order)

        params, cov_matrix = curve_fit(poly_fit, x_array, y_array, p0=[1] * (order + 1))
        # calculating chi-squared and red chi-squared for each polynomial fit
        χ2 = np.sum(((poly_fit(x_array, *params) - y_array) / yerr_array)**2)
        χ2_red = χ2 / 𝜈

        print(f"Polynomial fit of order {order}:\n chi-squared ={χ2:.3f} and reduced chi-squared = {χ2_red:.3f}")

        chi_squared_values.append(χ2)
        red_chi_squared_values.append(χ2_red)
    plt.plot(deg,red_chi_squared_values)
    plt.ylabel("Reduced chi-squared")
    plt.xlabel("Polynomial degree")
    plt.title("Exercise 2")
    plt.show()

    print("\nReduced chi-squared is closest to 1 after polynomial of order 4. \nWe don't want to overfit therefore order 4 is the best fit.\n")


# Exercise 3. Chi-squared testing a distribution.

# Determine whether the data from the experiment 
# conforms to a Poisson distribution using a reduced chi-squared test.
def P_poisson_μ(trials, x, μ):
    return trials * np.exp(-μ) * (μ**x / factorial(x))

def Exercise3():
    count_max = np.max(radioactive_data)
    count_min = np.min(radioactive_data)
    # range
    x = np.arange(count_min, count_max + 1)

    # statistics
    μ = np.mean(radioactive_data)
    std = np.sqrt(μ)
    trials = len(radioactive_data)

    poisson = P_poisson_μ(trials, x, μ)

    rebinned_poisson = poisson.reshape(12,3).sum(axis=1)
    # print(f"{rebinned_poisson = }")

    # making our histogram
    print("I really struggled with the plotting task in exercise 3... the bin edges where a big issue.")
    plt.grid(linestyle=':')
    plt.hist(x, [27, 32.8, 38.6, 44.4, 62], alpha=0.3)
    plt.bar(x[::3], rebinned_poisson)
    plt.xlabel('counts')
    plt.ylabel('frequency')
    plt.title('Exercise 3')
    plt.show()
    
    𝜈 = 6 - 2
    print(f" degrees of freedom: {𝜈 =}")




#~* FUNCTION CALLS *~#
polynomial_data, radioactive_data = read_files()

# Exercise 1.
red_chi2, p_arrays, red_chi2_values, p_max_values = reduced_Chi2(degrees_freedom)
Task1(degrees_freedom, red_chi2, p_arrays, red_chi2_values, p_max_values)
P_values = Task2(degrees_freedom, red_chi2, red_chi2_values)
Task3(degrees_freedom, red_chi2, p_arrays, red_chi2_values, p_max_values, P_values)

# Exercise 2.
deg = [2, 3, 4, 5, 6, 7, 8]
Exercise_2(polynomial_data, deg)

# Exercise 3.
Exercise3(radioactive_data)
# ran out of stamina :(
