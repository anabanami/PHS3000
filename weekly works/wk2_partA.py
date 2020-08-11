# PHS3000 - Uncertainties and anvanced fitting -Part A
# Exploring conditional statements
# Ana Fabela, 10/08/2020

import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS3000 as spa
plt.rcParams['figure.dpi'] = 150

# Part A - Conditional slicing
# # f strings are rad
print(f"{1 > 2=}") #False!
print(f"{10 <= 20=}") #True
print(f"{1 != 2=}") #True
print(f"{5 == (3 + 2)=}")
unorthodox = "greetings" + " " + "world" # I can add strings to make other strings
# print(unorthodox - "greetings" == " world") # Substraction is not supported: error

print(unorthodox == "hello world")#####FALSE#######

print(f"{0.5 == 0.499999999999999=}") #False, because Floating-point maths
print(f"{np.e**np.pi - np.pi=}") 

# let's continue with our script
a = 7
b = 3
print("Comparison using variables")
print(f"{b < a=}") #this is True
print(f"{a == 4=}") #this is False
print("Comparison using equations")
print(f"{(2 * b) < a=}") #this is True
print(f"{a == 4=}") #this is False

# slicing np arrays
c = np.array([
        [1.1, 2.1, 3.1],
        [1.2, 2.2, 3.2], 
        [1.3, 2.3, 3.3],
        [1.4, 2.4, 3.4],
        [1.5, 2.5, 3.5]
])

# 1D numpy arrays for independent (x), dependent variables (y) and uncertainty in (y)
x = c[:,0]
y=c[:,1]
u_y = c[:,2]
print(f"\n{x=}\n{y=}\n{u_y=}\n")
#an array vs a float?
print(f"{x < 1.3=}") # checks every item in array and compares to 1.3 
# returns array of Booleans

# conditional slicing, so clever!
x_subset = x[x<1.3]
print(f"{x_subset=}")
y_subset = y[x<1.3]
u_y_subset = u_y[x<1.3]
print(f"\n{y_subset=}\n{u_y_subset=}\n")

# when slicing numpy arrays we can use
# AND: (x>2)&(x<=7)
# OR: (x>2)|(x<=7)
# NOT: ~(x>2)
subset_x, subset_y, subset_u_y = x[~((1.1<x)&(x<1.5))], y[(y<2.2)|(y>2.4)], u_y[(~(u_y>3.1)|~(u_y<3.5))]
print(f"\n{subset_x=}\n{subset_y=}\n{subset_u_y=}\n")



