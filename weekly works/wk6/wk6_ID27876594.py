# PHS3000
# Week 6 - Poisson distribution for radioactive decay -
# Ana Fabela Hinojosa, 12/09/2020
import monashspa.PHS3000 as spa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize

plt.rcParams['figure.dpi'] = 200

radioactivity_data = spa.betaray.read_data(r'radioactive_counts.csv')
radioactivity_data = [x for sublist in radioactivity_data for x in sublist]

radioactive_counts = []
for string in radioactivity_data:
    count = int(string)
    radioactive_counts.append(count)

print(radioactive_counts)

def gaussian(x, amp, cen, sigma, c):
    return amp*(np.exp((-1.0/2.0)*(((x-cen)/sigma)**2))) + c

def gauss_fit(f, x, y):
    # 1 peak gaussian fit
    # unpack into popt, pcov
    popt_gauss, pcov_gauss = scipy.optimize.curve_fit(gaussian, x, y, p0=[amp, cen, sigma, c])
    perr_gauss = np.sqrt(np.diag(pcov_gauss))
    pars_gauss = popt_gauss[0:3]
    gauss_peak = gaussian(x, *pars_gauss, c)
    # return fit parameters
    return pars_gauss, gauss_peak, c

# STATISTICS
mean = np.mean(radioactive_counts)
std = np.std(radioactive_counts)
std_err = std / np.sqrt(len(radioactive_counts))

print(f"{mean = :.2f},{std = :.2f}, {std_err = :.2f}")

# Does the standard deviation agree with what you would expect for a Poisson distribution? 

std_poisson = np.sqrt(mean)
# print(f"comparison:{std = :.2f}, {std_poisson = :.2f}")
u_std_poisson = (1 / 2) * std_poisson * std_err / mean

print(f"{std_poisson = :.2f} ± {u_std_poisson:.2f}")
print("It is close, but it disagrees with the standard deviation.")

# Is it valid to approximate the Poisson distribution with a Gaussian in  #
# this experiment, based on the data?  Why?

msg = """The standard deviation (assuming that the data follows a gaussian distribution)
    is smaller but not too different from the calculated standard
    deviation assuming the process follows a poisson distribution."""

print(' '.join(msg.split()))


# number of bins
K = np.int(np.ceil(np.sqrt(len(radioactive_counts))))
print(f"number of bins {K=}")

# making our histogram
counts, bin_edges = np.histogram(radioactive_counts, bins=K)

# handmade histogram

plt.grid(linestyle=':')
plt.xticks(bin_edges[:-1], fontsize=8)
plt.bar(bin_edges[:-1], counts, color='#F8C1BB', alpha=0.5)
plt.xlabel('radioactive counts')
plt.ylabel('data points')
plt.title('Radioactivity data')

#first guess for gaussian parameters
amp, cen, sigma, c = mean, 37.25, std, 0

pars_gauss, gauss_peak, c = gauss_fit(gaussian, bin_edges[:-1], counts)

plt.plot(counts, gauss_peak, "y")
plt.fill_between(counts, gauss_peak.min(), gauss_peak, facecolor="yellow", alpha=0.5)
plt.show()