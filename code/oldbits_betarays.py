# 92
# # plot
# plt.figure()
# plt.errorbar(
#             p_rel, corrected_count, xerr=u_p_rel, yerr=u_corrected_count,
#             marker="None", ecolor="m", label=r"$n(p)_{corrected}$", color="g", barsabove=True
# )

# plt.title(r"$\beta^{-}$ particle momentum spectrum")
# plt.xlabel("p [mc]")
# plt.ylabel("n(p)")
# plt.legend()
# spa.savefig('count_vs_momentum_no_background_error.png')
# # plt.show()

# 113
# n_p_rel, w_rel = n(p_rel[:22]) #  call and unpack n(p)

# 130
# # equation (3) in script
# N = n_p_rel * dp_rel 

# # plot
# plt.figure()
# plt.plot(
#         p_rel[:22], N, marker="None",
#         linestyle="-"
# )
# plt.title("Kurie relation")
# plt.xlabel("p [mc]")
# plt.ylabel("n(p)dp")
# spa.savefig('Kurie_plot.png')
# plt.show()


# 140
# correcter_count = corrected_count[4:23] / lens_current[4:23]
# u_correcter_count = np.sqrt((u_corrected_count / corrected_count[4:23])**2 + (u_lens_current[4:23] / lens_current[4:23])**2)
# print(f"{u_correcter_count=}")

# plot
# plt.figure()
# plt.errorbar(
#             p_rel[4:23], correcter_count, xerr=u_p_rel[4:23], yerr=u_correcter_count,
#             marker="None", ecolor="m", label=r"$n(p)_{corrected}$", color="g", barsabove=True
# )

# plt.title(r"$\beta^{-}$ particle momentum spectrum")
# plt.xlabel("p [mc]")
# plt.ylabel("n(p)")
# plt.legend()
# spa.savefig('count_vs_momentum_no_background_error_corrected_resolution.png')
# plt.show()

# 147
# initial slice [:23]
# second slice [8:18]

# 201
# this clips negative counts which are non physical
# corrected_count = corrected_count.clip(min=0)

# # LINEARISED KURIE 
# y = np.sqrt(corrected_count[8:18] / (p_rel[8:18] * x * interpolated_fermi(p_rel[8:18])))
# # regularising y to avoid zero u_y 
# y_regularised = np.sqrt(corrected_count[8:18].clip(min=1) / (p_rel[8:18] * x * interpolated_fermi(p_rel[8:18])))
# u_y = (y_regularised / 2) * np.sqrt((u_corrected_count / corrected_count[8:18].clip(min=1))**2 + (2 * (u_p_rel[8:18] / p_rel[8:18])**2) + (u_interpolated_fermi / interpolated_fermi(p_rel[8:18]))**2)

# 287
#  wrong comparison:
# print(f"EXPECTED RESULT {theory_w_0_rel = }")
# print(f"post-optimisation result  {opt_w_0 = } ± {u_opt_w_0}\n")
# print(f"non-relativistic w_0 = {opt_w_0 * rel_energy_unit / MeV} ± {u_opt_w_0 * rel_energy_unit / MeV}\n")


#  292 
# print("relativistic units:")
# print(f"EXPECTED RESULT {theory_T_rel = :.3f}")
# print(f"post-optimisation result {opt_T_rel = :.3f} ± {u_opt_T_rel:.3f}\n")


#? (pre-optimisation)
# # LINEARISED KURIE WITH RESOLUTION CORRECTION !!!
# y = np.sqrt(correct_count[8:18] / (p_rel[8:18] * x * interpolated_fermi(p_rel[8:18])))
# u_y = (y / 2) * np.sqrt((u_correct_count[8:18] / correct_count[8:18].clip(min=1))**2 + (2 * (u_p_rel[8:18] / p_rel[8:18])**2) + (u_interpolated_fermi / interpolated_fermi(p_rel[8:18]))**2)
# fit_results = spa.linear_fit(x, y, u_y=u_y)
# # making our linear fit with one sigma uncertainty
# y_fit = fit_results.best_fit
# u_y_fit = fit_results.eval_uncertainty(sigma=1)
# # calculating values from fit results
# fit_parameters = spa.get_fit_parameters(fit_results)
# # print(f"{fit_parameters=}")
# # using our results to find w_0
# K_2 = - fit_parameters["slope"]
# u_K_2 = fit_parameters["u_slope"]
# intercept = fit_parameters["intercept"]
# u_intercept = fit_parameters["u_intercept"] 
# w_0 = intercept / K_2
# u_w_0 = np.sqrt((u_K_2 / K_2)**2 + (u_intercept / intercept)**2) * w_0
# # print(f"linear fit gradient: {K_2 = :.3f}± {u_K_2:.3f}")
# # print(f"linear fit intercept: {intercept = :.3f}\n")

# # using our results to find T
# # pre-optimisation result 
# T_rel = w_0 - 1
# u_T_rel = T_rel * (u_w_0 / w_0)

# # RESULT in SI units
# T_SI = T_rel * rel_energy_unit / MeV
# u_T_SI = T_rel * (u_w_0 / w_0) * rel_energy_unit / MeV
# # print(f"\nEXPECTED RESULT T = {theory_T / MeV :.3f} MeV")
# # print(f"(pre-optimisation) T = {T_SI:.3f} ± {u_T_SI:.3f} MeV\n")
# #?
# ############################ plots ############################
# plt.figure()
# plt.errorbar(
#             x, y, xerr=u_p_rel[8:18], yerr=u_y,
#             marker="None", linestyle="None", ecolor="m", 
#             label=r"$y = (\frac{n}{p w G})^{\frac{1}{2}}$", color="g", barsabove=True
# )
# plt.plot(
#         x, y_fit, marker="None",
#         linestyle="-", 
#         label="linear fit"
# )
# plt.fill_between(
#                 x, y_fit - u_y_fit,
#                 y_fit + u_y_fit,
#                 alpha=0.5,
#                 label="uncertainty in linear fit"
# )
# plt.title("Linearised Kurie data")
# plt.xlabel(r"$w [mc^{2}]$")
# plt.ylabel(r"$\left ( \frac{n}{p w G} \right )^{\frac{1}{2}}$", rotation=0, labelpad=18)
# plt.legend()
# # spa.savefig('Kurie_linear_data_plot_.png')
# # plt.show()

# ############################# Linear fit #######################
# ##########################Linear fit residuals##################
# linear_residuals = y_fit - y # linear residuals

# # plot
# plt.figure()
# plt.errorbar(
#             x, linear_residuals, xerr=u_p_rel[8:18], yerr=u_y,
#             marker="o", ecolor="m", linestyle="None",
#             label="Residuals (linearised data)"
# )
# plt.plot([x[0], x[-1]], [0,0], color="k")
# plt.title("Residuals: linearised Kurie data")
# plt.xlabel(r"$w [mc^{2}]$")
# plt.ylabel(r"$\left ( \frac{n}{p w G} \right )^{\frac{1}{2}}$", rotation=0, labelpad=18)
# plt.legend()
# # spa.savefig('linear_residuals_Kurie_linear_data.png')
# # plt.show()
# ##########################Linear fit residuals#################