# PHS3000
# Advanced fitting with python
# Ana Fabela, 10/08/2020

import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS3000 as spa
plt.rcParams['figure.dpi'] = 150

# *~playing around~*
# f strings are rad
# print(f"{1 > 2=}") #False!
# print(f"{10 <= 20=}") #True
# print(f"{1 != 2=}") #True
# print(f"{5 == (3 + 2)=}")
# unorthodox = "greetings" + " " + "world" # I can add strings to make other strings
# # print(unorthodox - "greetings" == " world") # Substraction is not supported: error

# print(unorthodox == "hello world")#####FALSE#######

# print(f"{0.5 == 0.499999999999999=}") #False, because Floating-point maths
# print(f"{np.e**np.pi - np.pi=}") 

# # let's continue with our script
# a = 7
# b = 3
# print("Comparison using variables")
# print(f"{b < a=}") #this is True
# print(f"{a == 4=}") #this is False
# print("Comparison using equations")
# print(f"{(2 * b) < a=}") #this is True
# print(f"{a == 4=}") #this is False

# # slicing np arrays
# c = np.array([
#         [1.1, 2.1, 3.1],
#         [1.2, 2.2, 3.2], 
#         [1.3, 2.3, 3.3],
#         [1.4, 2.4, 3.4],
#         [1.5, 2.5, 3.5]
# ])

# # 1D numpy arrays for independent (x), dependent variables (y) and uncertainty in (y)
# x = c[:,0]
# y=c[:,1]
# u_y = c[:,2]
# print(f"\n{x=}\n{y=}\n{u_y=}\n")
# #an array vs a float?
# print(f"{x < 1.3=}") # checks every item in array and compares to 1.3 
# # returns array of Booleans

# # conditional slicing, so clever!
# x_subset = x[x<1.3]
# print(f"{x_subset=}")
# y_subset = y[x<1.3]
# u_y_subset = u_y[x<1.3]
# print(f"\n{y_subset=}\n{u_y_subset=}\n")

# # when slicing numpy arrays we can use
# # AND: (x>2)&(x<=7)
# # OR: (x>2)|(x<=7)
# # NOT: ~(x>2)
# subset_x, subset_y, subset_u_y = x[~((1.1<x)&(x<1.5))], y[(y<2.2)|(y>2.4)], u_y[(~(u_y>3.1)|~(u_y<3.5))]
# print(f"\n{subset_x=}\n{subset_y=}\n{subset_u_y=}\n")

################################Nuclear decay#########################################
data = spa.tutorials.fitting.part_b_data
# print(data)
time = data[:, 0] # first column (time elapsed in seconds [s])
A = data[:, 1] # second column (A is activity [dimensionless])
u_A = data[:, 2] # third column (uncertainty in activity [dimensionless])

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


# creating figure environment
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

plt.xlabel("t [s]")
plt.ylabel("A [arb]")
# # plt.xlim([0, 2])
# # plt.ylim([0, 3])
plt.legend()
spa.savefig('nonlinear_silver_decay_data.png')
# # showmewhatchugot
plt.show()

################################Nuclear decay#########################################
# plt.figure(2)
# plt.title("Figure 2: Linear plot of pendulum data")
# plt.errorbar(
#             length,avg_period_sq, xerr=u_length,
#             yerr=u_avg_period_sq,
#             marker="o", linestyle="None", color="black",
#             label="pendulum data"
# )
# plt.plot(
#         length, y_fit, marker="None",
#         linestyle="-", color="black",
#         label="linear fit"
# )
# plt.fill_between(
#                 length,y_fit - u_y_fit,
#                 y_fit + u_y_fit, 
#                 color="lightgrey",
#                 label="uncertainty in linear fit"
# )
