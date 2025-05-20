import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import norm

# Load the HDF5 file
plot_dir = "plot"
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_hdf("/eos/user/t/tihsu/database/EGM_Scouting_Trig_Study/output.h5")

# Select entries that pass PFScouting_JetHT
df_base = df[df['PFScouting_JetHT'] == 1.0]

# Define probe pt bins and eta regions for legend
pt_bins = np.array([0, 10, 15, 20, 25, 30, 35, 50, 80, 120, 200])  # 0 to 100 GeV, 5 GeV bins
eta_bins = [(0, 1.5), (1.5, 2.5)]

# Fit function: Gaussian (signal) + Exponential (background)
def gaus_exp(x, A, mu, sigma, B, k):
    return A * norm.pdf(x, mu, sigma) + B * np.exp(-k * x)

# Prepare plot
plt.figure(figsize=(10, 6))
all_efficiencies = dict()

for eta_min, eta_max in eta_bins:
    eff_pts = []
    pt_centers = []

    for i in range(len(pt_bins) - 1):
        pt_low, pt_high = pt_bins[i], pt_bins[i + 1]
        pt_center = 0.5 * (pt_low + pt_high)

        # Apply |eta| cut
        eta_mask = (np.abs(df_base["probe-eta"]) >= eta_min) & (np.abs(df_base["probe-eta"]) < eta_max)
        pt_mask = (df_base["probe-pt"] >= pt_low) & (df_base["probe-pt"] < pt_high)

        # Total (JetHT)
        df_total = df_base[eta_mask & pt_mask]
        # Pass (DoubleEG ∩ JetHT)
        df_pass = df_total[df_total["PFScouting_DoubleEG"] == 1.0]

        if len(df_pass) < 20 or len(df_total) < 20:
            continue

        # Fit both: denominator (total) and numerator (pass)
        for kind, df_fit, tag in [("Total", df_total, "total"), ("Pass", df_pass, "pass")]:
            mass = df_fit["inv-mass"].values
            hist_vals, bin_edges = np.histogram(mass, bins=20, range=(70, 110))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            p0 = [max(hist_vals), 90, 2, min(hist_vals), 0.05]

            try:
                popt, _ = curve_fit(gaus_exp, bin_centers, hist_vals, p0=p0)
                A, mu, sigma, B, k = popt
                signal_yield = A * sigma * np.sqrt(2 * np.pi)

                # Save results
                if kind == "Total":
                    signal_total = signal_yield
                else:
                    signal_pass = signal_yield

                # Plot and save fit
                plt_fit = plt.figure(figsize=(6, 4))
                plt.hist(mass, bins=60, range=(70, 110), alpha=0.5, label="Data")
                x_fit = np.linspace(70, 110, 1000)
                plt.plot(x_fit, gaus_exp(x_fit, *popt), 'r-', label='Fit: G+Exp')
                plt.title(f"{kind}: pT ∈ [{pt_low}, {pt_high}] GeV, |η| ∈ [{eta_min}, {eta_max}]")
                plt.xlabel("Invariant Mass [GeV]")
                plt.ylabel("Entries")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"fit_{tag}_pt{int(pt_low)}_abseta{eta_min:.1f}_to_{eta_max:.1f}.png"))
                plt.close(plt_fit)

            except RuntimeError:
                signal_total, signal_pass = 0, 0
                break  # skip this bin if fit fails

        if signal_total > 0:
            eff = signal_pass / signal_total
            pt_centers.append(pt_center)
            eff_pts.append(eff)
    all_efficiencies[(eta_min, eta_max)] = (pt_centers, eff_pts)


plt.figure(figsize=(8, 6))

for (eta_min, eta_max), (pt_centers, eff_pts) in all_efficiencies.items():
    label = f"{eta_min:.1f} ≤ |η| < {eta_max:.1f}"
    plt.plot(pt_centers, eff_pts, marker='o', label=label)

plt.xlabel("Probe $p_T$ [GeV]", fontsize=14)
plt.ylabel("Trigger Efficiency (fit-based)", fontsize=14)
plt.title("Efficiency vs $p_T$ from S(G) / S(G)", fontsize=15)
plt.legend(title=r"$|\eta|$ bin")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "fitted_efficiency_curve_abseta.pdf"))
plt.show()
