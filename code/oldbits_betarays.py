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