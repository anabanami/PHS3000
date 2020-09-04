# PHS3000
# Week 5 - STATISTICS -
# Ana Fabela Hinojosa, 1/09/2020
import monashspa.PHS3000 as spa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

noise_data = spa.betaray.read_data(r'noise_data.csv')
pendulum_data = spa.betaray.read_data(r'pendulum_data.csv')

# Exercise 1
# Rewrite the following results in the clearest form using correct significant figures
print(f"\nExercise 1\n")
a = 3.323
u_a = 1.4 

b = 1234567 / 1e6
u_b = 54321 / 1e6

c = 5.33
u_c = 3.21 * 1e-2

d = 0.000000538 / 1e-7
u_d = 0.00000003 / 1e-7

e = 5.03
u_e = 0.04329

f = 1.5432
u_f = 1

g = 3.267
u_g = 42 / 1e2

print(f"(a) {a:.1f} ± {u_a:.1f} mm")
print(f"(b) ({b:.2f} ± {u_b:.2f}) ×10^6 s")

print(f"(c) ({c:.2f} ± {u_c:.2f}) ×10^-7 m")

print(f"(d) ({d:.1f} ± {u_d:.1f}) ×10^-7 m")
print(f"(e) {e:.2f} ± {u_e:.2f} m")
print(f"(f) {f:.0f} ± {u_f} s")
print(f"(g) ({g:.2f} ± {u_g:.2f}) ×10^3 g * cm / s")

# Exercise 2
# (a) The best estimate fo the period of the pendulum, p.
# What parameter should you use for the measure of the uncertainty in the period? Calculate this uncertainty.

print(f"\nExercise 2\n")

#first we convert our array to a numpy nd array
pendulum_data = np.asarray(pendulum_data, dtype=np.float32)

# calculating our stats
μ = np.mean(pendulum_data)
𝝈 = np.std(pendulum_data)
std_err = 𝝈 / np.sqrt(len(pendulum_data))

print(f"The best estimate for the period of the pendulum is the mean period {μ =:.3f} s")
print(f"the best measure of the uncertainty in the period is the standard error = {std_err:.3f} s")
# (b) What is the uncertainty, Δ𝑝,  in any individual measurement of the pendulum’s period?
print(f"the uncertainty in any individual measurement of the the period (Δ𝑝) is the standard deviation {𝝈 = :.2f} s\n")



# Exercise 3: Histogram plots of the pendulum data

# (a) Using the data for the pendulum period (previous question), create a histogram of 
# the data using the “best estimate” rule for randomly distributed data to set the number of bins.
print(f"\nExercise 3\n")
print(f"Check them histograms!\n")

# number of bins
K = np.int(np.ceil(np.sqrt(len(pendulum_data))))
print(f"number of bins {K=}")

# making our histogram
counts, bin_edges = np.histogram(pendulum_data, bins=K)
print(f"{counts = }, length of list: {len(counts)}")
print(f"{bin_edges[:-1]=}, length of list: {len(bin_edges[:-1])}")

# handmade histogram
# plt.figure(1)
# plt.grid(linestyle=':')
# plt.xticks(bin_edges[:-1], fontsize=8)
# plt.bar(bin_edges[:-1], counts,width=0.016, color='#F8C1BB', alpha=0.5)
# plt.xlabel('period / s')
# plt.ylabel('counts')
# plt.title('Pendulum period data')
# plt.show()

# using plt.hist()
plt.figure(2)
plt.grid(linestyle=':')
plt.hist(pendulum_data, bins=K, width=0.016, color='#00BFF7', alpha=0.5)
plt.xlabel('period / s')
plt.ylabel('counts')
plt.title('Pendulum period data')
# plt.show()

# (b) Examine the effect of changing the number of bins from this value by changing the bin number 
# by integer values in either direction. Note the variation in shape.

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(pendulum_data, bins=K-2,color='#4739E9')
axs[0, 0].set_title(f'K = {K-2}')
axs[0, 1].hist(pendulum_data, bins=K*2, color='#00BFF7')
axs[0, 1].set_title(f'K = {K*2}')
axs[1, 0].hist(pendulum_data, bins=K*4, color='#EC92D7')
axs[1, 0].set_title(f'K = {int(K*4)}')
axs[1, 1].hist(pendulum_data, bins=K*K, color='#1BAA53')
axs[1, 1].set_title(f'K = {int(K*K)}')

for ax in axs.flat:
    ax.set(xlabel='period / s', ylabel='counts')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
# plt.show()

# Exercise 4: Histograms to reveal features in data

# (a) Assuming normally distributed data, produce a histogram of the data contained in the data set noise_data

print(f"\nExercise 4\n")

#first we convert our array to a numpy nd array
noise_data = np.asarray(noise_data, dtype=np.float32)

K = np.int(np.ceil(np.sqrt(len(noise_data))))
print(f"number of bins for noise data {K=}")

# making our histogram
counts, bin_edges = np.histogram(noise_data, bins=K)

plt.figure(5)
plt.grid(linestyle=':')
plt.hist(noise_data, bins=K, color='g', alpha=0.5)
plt.xlabel('x')
plt.ylabel('counts')
plt.title('Noise data')
# plt.show()


# (b) Investigate re-binning the data using different scales. Based on your investigations, what conclusions can you make concerning the data?

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(noise_data, bins=int(K*2), color='blue', alpha=0.5)
axs[0, 0].set_title(f'K = {K*2}')
axs[0, 1].hist(noise_data, bins=int(K*10), color='orange', alpha=0.5)
axs[0, 1].set_title(f'K = {K*10}')
axs[1, 0].hist(noise_data, bins=int(K / 5), color='green', alpha=0.5)
axs[1, 0].set_title(f'K = {int(K / 5)}')
axs[1, 1].hist(noise_data, bins=int(K / 10), color='red', alpha=0.5)
axs[1, 1].set_title(f'K = {int(K / 10)}')
# plt.xticks(fontsize=5)

for ax in axs.flat:
    ax.set(xlabel='x', ylabel='counts')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()

print("The distribution appears to resolve to a normal distribution.")
print("Increasing the number of bins promotes a strange effect: the most common values dominate the data")
print("creating very high count peaks... In this scenario: the plots with less bins show more concise information.")
print("Never the less the square root rule is superior for data presentation and clarity.")