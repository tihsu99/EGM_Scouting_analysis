import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import norm
import argparse
import ROOT
import cmsstyle as CMS
from scipy.special import erf
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
import glob
import array


CMS.SetExtraText("Preliminary")
CMS.SetEnergy("13.6")
CMS.SetLumi("27.8")

hex_colors = [
    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
    "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"
]
default_colors = []

for i, hex_color in enumerate(hex_colors):
    color_id = ROOT.TColor.GetColor(hex_color)
    default_colors.append(color_id)


def fit_hist(hist, xmin, xmax, fig_name, era, lumi):
  # --- Extract data from histogram ---
  x_vals = []
  y_vals = []
  y_errs = []

  for i in range(1, hist.GetNbinsX() + 1):
    x = hist.GetBinCenter(i)
    y = hist.GetBinContent(i)
    err = hist.GetBinError(i)
    if (y > 0) and (x > xmin) and (x < xmax):
        x_vals.append(x)
        y_vals.append(y)
        y_errs.append(err if err > 0 else np.sqrt(y))

  x_vals = np.array(x_vals)
  y_vals = np.array(y_vals)
  y_errs = np.array(y_errs)

  total = np.sum(y_vals)

  # ------ Double Crystal Ball Func ----- #
  def double_crystal_ball(x, mean, sigma, alpha_l, n_l, alpha_r, n_r, amp):
    x = np.array(x)
    t = (x - mean) / sigma
    
    # Safety guards
    alpha_l = np.abs(alpha_l) if abs(alpha_l) > 1e-6 else 1e-6
    alpha_r = np.abs(alpha_r) if abs(alpha_r) > 1e-6 else 1e-6
    n_l = max(n_l, 1.01)
    n_r = max(n_r, 1.01)
    amp = max(amp, 0)

    # Left tail constants
    A_l = (n_l / alpha_l)**n_l * np.exp(-0.5 * alpha_l**2)
    B_l = n_l / alpha_l - alpha_l

    # Right tail constants
    A_r = (n_r / alpha_r)**n_r * np.exp(-0.5 * alpha_r**2)
    B_r = n_r / alpha_r - alpha_r

    # Normalization constant (analytic)
    C_l = n_l / alpha_l / (n_l - 1) * np.exp(-0.5 * alpha_l**2)
    C_r = n_r / alpha_r / (n_r - 1) * np.exp(-0.5 * alpha_r**2)
    D = np.sqrt(np.pi / 2) * (1 + erf(alpha_l / np.sqrt(2))) + np.sqrt(np.pi / 2) * (1 + erf(alpha_r / np.sqrt(2)))

    N = 1.0 / (sigma * (C_l + C_r + D))

    # Piecewise definition
    result = np.zeros_like(t)

    # Gaussian core
    core_mask = (t > -alpha_l) & (t < alpha_r)
    result[core_mask] = np.exp(-0.5 * t[core_mask]**2)

    # Left power tail
    left_mask = t <= -alpha_l
    result[left_mask] = A_l * (B_l - t[left_mask])**(-n_l)

    # Right power tail
    right_mask = t >= alpha_r
    result[right_mask] = A_r * (B_r + t[right_mask])**(-n_r)

    return amp * N * result

  # --- Crystal Ball function ---
  def crystal_ball(x, alpha, n, mean, sigma, amp):
    x = np.array(x)
    t = (x - mean) / sigma
    abs_alpha = np.abs(alpha)

    # Constants with safety
    if abs_alpha < 1e-3:
        abs_alpha = 1e-3
    if n < 1.01:
        n = 1.01

    if amp < 0:
        amp = 0

    A = (n / abs_alpha)**n * np.exp(-0.5 * abs_alpha**2)
    B = n / abs_alpha - abs_alpha
    C = n / abs_alpha / (n - 1) * np.exp(-0.5 * abs_alpha**2)
    D = np.sqrt(np.pi / 2) * (1 + erf(abs_alpha / np.sqrt(2)))
    N = 1.0 / (sigma * (C + D))

    # Build piecewise function
    result = np.zeros_like(t)
    mask = t > -abs_alpha
    result[mask] = np.exp(-0.5 * t[mask]**2)
    result[~mask] = A * np.power(B - t[~mask], -n)
    return amp * N * result

  # --- Full model: background + Gaussian + Crystal Ball ---
  def model_gaus(x,
          bkg_amp, bkg_slope,
          gaus_amp, gaus_mean, gaus_sigma):

    background = bkg_amp * np.exp(bkg_slope * x)
    gaussian = gaus_amp * np.exp(-(x - gaus_mean)**2 / (2 * gaus_sigma**2))
    return background + gaussian

  def model(x,
          bkg_amp, bkg_slope,
          dcb_amp, dcb_alpha_l, dcb_n_l, dcb_alpha_r, dcb_n_r, dcb_mean, dcb_sigma):

    background = bkg_amp * np.exp(bkg_slope * x)
    dcb = double_crystal_ball(x, dcb_mean, dcb_sigma, dcb_alpha_l, dcb_n_l, dcb_alpha_r, dcb_n_r, dcb_amp)
    return background + dcb

  p0 = [total*0.1, 0.0,        # Background: amp, slope
      total*0.9, 1.5, 3.0, 1.5, 3.0, (xmax + xmin) / 2, 3.0]  # Crystal Ball: amp, alpha, n, mean, sigma

  # --- Fit with scipy ---
  popt, pcov = curve_fit(model, x_vals, y_vals, p0=p0, sigma=y_errs, absolute_sigma=True, maxfev=10000, bounds = (
    [0, -10,     0,   0.5, 1.01, 0.5, 1.01, xmin, 0.1],  # Lower bounds
    [np.inf, 0, np.inf, 5.0, 20.0, 5.0, 20.0, xmax, 10.0]  # Upper bounds
  ))

  # --- Plot results ---
  x_fit = np.linspace(min(x_vals), max(x_vals), 1000)
  y_fit = model(x_fit, *popt)

  fig, ax = plt.subplots()



  ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='ko', label='Data', markersize=3)
  ax.plot(x_fit, y_fit, 'm-', label='Signal + Background')
  ax.plot(x_fit, popt[0] * np.exp(popt[1] * x_fit), 'brown', label='Background')
  ax.plot(x_fit, double_crystal_ball(x_fit, *popt[7:9], *popt[3:7], popt[2]), 'cyan', label='Signal(DCB)')

  # Calculate the background curve
  background_curve = popt[0] * np.exp(popt[1] * x_fit)

  # Calculate the signal curve (Crystal Ball)
  signal_curve = double_crystal_ball(x_fit, *popt[7:9], *popt[3:7], popt[2])
  # Perform the numerical integration (Trapezoidal rule)
  background_integral = np.trapz(background_curve, x_fit)
  signal_integral = np.trapz(signal_curve, x_fit)

  ax.set_xlabel("Dielectron mass [GeV]")
  ax.set_ylabel("Events / 1 GeV")
  ax.legend()
  ax.grid()

  hep.cms.text("Preliminary", loc=2, ax=ax, fontsize=12)
  hep.cms.lumitext(f"Run3 {era} {lumi} $\\mathrm{{fb}}^{{-1}}$ (13.6 TeV)", ax=ax)


  plt.text(0.05, 0.8, f"background integral: {background_integral:.0f}", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')
  plt.text(0.05, 0.75, f"signal integral: {signal_integral:.0f}", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')
  plt.text(0.05, 0.7, f"mean: {popt[5]:.1f} GeV", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')
  plt.text(0.05, 0.65, f"$\\sigma_{{cb}}$: {popt[6]:.1f} GeV", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')
  plt.text(0.05, 0.6, f"relative width: {100*popt[6]/popt[5]:.1f} %", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left', color='black')


  plt.tight_layout()
  plt.savefig(f"{fig_name}.png")
  plt.show()
  plt.close()

  # --- Print fit results ---
  param_names = [
    "bkg_amp", "bkg_slope",
    "cb_amp", "cb_alpha_l", "cb_n_l", "cb_alpha_r", "cb_n_r", "cb_mean", "cb_sigma"
  ]


  print("Fit results:")
  output_dict = dict()
  for name, val in zip(param_names, popt):
    print(f"{name:>10} = {val:.6f}")
    output_dict[name] = val

  output_dict["signal_integral"] = signal_integral

  return output_dict



def tag_and_probe(df_base, pt_bins, eta_bins, Target_DST, fit):
    histograms = dict()

    for eta_min, eta_max in eta_bins:
        hist_pass = ROOT.TH1F(f"pass_eta_{eta_min}_{eta_max}", f"Pass: {eta_min} < |η| < {eta_max}", len(pt_bins)-1, np.array(pt_bins, dtype=np.float64))
        hist_total = ROOT.TH1F(f"total_eta_{eta_min}_{eta_max}", f"Total: {eta_min} < |η| < {eta_max}", len(pt_bins)-1, np.array(pt_bins, dtype=np.float64))

        for i in range(len(pt_bins) - 1):
            pt_low, pt_high = pt_bins[i], pt_bins[i + 1]

            # Apply |eta| cut
            eta_mask = (np.abs(df_base["probe-eta"]) >= eta_min) & (np.abs(df_base["probe-eta"]) < eta_max)
            pt_mask = (df_base["probe-pt"] >= pt_low) & (df_base["probe-pt"] < pt_high)

            df_total = df_base[eta_mask & pt_mask]
            df_pass = df_total[df_total[Target_DST] == 1.0]

            if not fit:
                signal_pass = df_pass.shape[0]
                signal_total = df_total.shape[0]

            else:
                bin_edges = np.arange(70, 110, 1)
                # Compute the histogram
                counts_pass, bin_edges_pass = np.histogram(df_pass["inv-mass"], bins=bin_edges)
                counts_total, bin_edges_total = np.histogram(df_total["inv-mass"], bins=bin_edges)
                bin_edges = bin_edges_pass
                # Compute bin widths
                bin_widths = np.diff(bin_edges)
                bin_edges = array.array('f', bin_edges)
                hist_pass_pt = ROOT.TH1F("hist", "title", len(bin_edges)-1, bin_edges)
                hist_total_pt = ROOT.TH1F("hist2", "title", len(bin_edges)-1, bin_edges)
                # Fill TH1 bin contents (skip underflow bin 0)
                for i_pass, count in enumerate(counts_pass):
                    hist_pass_pt.SetBinContent(i_pass + 1, count)
                for i_total, count in enumerate(counts_total):
                    hist_total_pt.SetBinContent(i_total + 1, count)

                xmin = 70
                xmax = 110

                if hist_pass_pt.Integral() > 1000:

                    pass_stat = fit_hist(hist_pass_pt, xmin, xmax, os.path.join(plot_dir, f"eta_{eta_min}_ptbin_{i}_pass"), "run3", 0)
                    total_stat = fit_hist(hist_total_pt, xmin, xmax, os.path.join(plot_dir, f"eta_{eta_min}_ptbin_{i}_total"), "run3", 0)

                    signal_pass = pass_stat["signal_integral"]
                    signal_total = total_stat["signal_integral"]
                else:
                    signal_pass = df_pass.shape[0]
                    signal_total = df_total.shape[0]

            print(signal_pass, signal_total)

            # Compute bin center and bin number
            pt_center = 0.5 * (pt_low + pt_high)
            bin_idx_pass = hist_pass.FindBin(pt_center)
            bin_idx_total = hist_total.FindBin(pt_center)

            # Set content and Poisson errors manually
            hist_pass.SetBinContent(bin_idx_pass, signal_pass)
            hist_pass.SetBinError(bin_idx_pass, np.sqrt(signal_pass))
 
            hist_total.SetBinContent(bin_idx_total, signal_total)
            hist_total.SetBinError(bin_idx_total, np.sqrt(signal_total))


        histograms[(eta_min, eta_max)] = (hist_pass, hist_total)

    return histograms


def Derive_Trigger_Efficiency(df, selection, xText, Target_DST, fit, pt_bins, postfix = ""):
    output = dict()

    df_base = df[selection]

    # Define probe pt bins and eta regions for legend
    eta_bins = [(0, 1.5)]

    for wp_idx, wp in enumerate(["Loose-noIso", "Medium-noIso", "Tight-noIso"]):
      wp_cut = wp_idx +1
      df_base_wp = df_base[df_base["probe-noiso-id"] >= wp_cut]
      output[wp] = tag_and_probe(df_base_wp, pt_bins, eta_bins, Target_DST, fit)

    c = None

    icolor = 0
    hist_eff = dict()
    for wp, histograms in output.items():
      hist_eff[wp] = dict()
      for (eta_min, eta_max), (hist_pass, hist_total) in histograms.items():
         hist_eff[wp][(eta_min, eta_max)]   = ROOT.TEfficiency(hist_pass, hist_total)

         x_axis = hist_pass.GetXaxis()
         nbinX  = x_axis.GetNbins()
         x_binnings = [x_axis.GetBinLowEdge(bin_+1) for bin_ in range(nbinX+1)]


         if c is None:
           c = CMS.cmsCanvas('', min(x_binnings), max(x_binnings), 0, 1.1, "scouting electron p_{T} [GeV]", "L1+HLT efficiency", square = CMS.kSquare, extraSpace=0.0, iPos=0)
           legend = CMS.cmsLeg(0.44, 0.2, 0.9, 0.4, textSize=0.03)

         CMS.cmsDraw(hist_eff[wp][(eta_min, eta_max)], 'P E', lcolor = default_colors[icolor], mcolor = default_colors[icolor], msize=1, lwidth=2, fstyle=0)
         legend.AddEntry(hist_eff[wp][(eta_min, eta_max)], f"{eta_min} < |#eta| < {eta_max}, {wp}", 'PL')

         icolor += 1

    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(62)  # Bold
    label.SetTextSize(0.03)
    label.SetTextAlign(11)  # Left-top corner

    DST_text = "Scouting double-EG trigger(2024)" if "DoubleEG" in Target_DST else "Scouting single-Photon trigger(2024)"
    label.DrawLatex(0.47, 0.42, DST_text)

    hdf = CMS.GetcmsCanvasHist(c)
    hdf.GetYaxis().SetLabelSize(0.04)
    hdf.GetYaxis().SetTitleOffset(1.2)
    hdf.GetYaxis().SetTitleSize(0.05)
    hdf.GetXaxis().SetLabelSize(0.04)
    hdf.GetXaxis().SetTitleOffset(1.2)
    hdf.GetXaxis().SetTitleSize(0.05)

    c.SetLogx()
    CMS.SaveCanvas(c, os.path.join(plot_dir, f'{Target_DST}_Efficiency{postfix}.png'), close = False)
    CMS.SaveCanvas(c, os.path.join(plot_dir, f'{Target_DST}_Efficiency{postfix}.pdf'))



def Derive_Trigger_Efficiency_orthogonal_2D(df, selection, xText, Target_DST, fit, pt_bins, postfix = ""):
    output = dict()
    df["leading-pt"]     = np.maximum(df["tag-pt"], df["probe-pt"])
    df["subleading-pt"]  = np.minimum(df["tag-pt"], df["probe-pt"])
    df_base = df[selection]

    # Define probe pt bins and eta regions for legend
    eta_bins = [(0, 1.5)]

    # Define 2D binning for leading/subleading pT
    pt_bins = np.array(pt_bins, dtype=float)
    n_pt = len(pt_bins) - 1

    for wp_idx, wp in enumerate(["Tight"]):
      wp_cut = wp_idx +1
      df_base_wp = df_base[df_base["probe-id"] >= wp_cut]
      for eta_min, eta_max in eta_bins:
        hist_pass = ROOT.TH2F(f"hist_pass_{wp}{postfix}", "", n_pt, pt_bins, n_pt, pt_bins)
        hist_total = ROOT.TH2F(f"hist_total_{wp}{postfix}", "", n_pt, pt_bins, n_pt, pt_bins)


        for i in range(len(pt_bins) - 1):
          for j in range(len(pt_bins) - 1):
            leading_pt_low, leading_pt_high = pt_bins[i], pt_bins[i + 1]
            subleading_pt_low, subleading_pt_high = pt_bins[j], pt_bins[j + 1]

            # Apply |eta| cut
            eta_mask = (np.abs(df_base["probe-eta"]) >= eta_min) & (np.abs(df_base["probe-eta"]) < eta_max) & (np.abs(df_base["tag-eta"]) >= eta_min) & (np.abs(df_base["tag-eta"]) >= eta_min) 
            pt_mask = (df_base["leading-pt"] >= leading_pt_low) & (df_base["leading-pt"] < leading_pt_high) & (df_base["subleading-pt"] >= subleading_pt_low) & (df_base["subleading-pt"] < subleading_pt_high)

            df_total = df_base[eta_mask & pt_mask]
            df_pass = df_total[df_total[Target_DST] == 1.0]

            if not fit:
                signal_pass = df_pass.shape[0]
                signal_total = df_total.shape[0]

            else:
                bin_edges = np.arange(70, 110, 1)
                # Compute the histogram
                counts_pass, bin_edges_pass = np.histogram(df_pass["inv-mass"], bins=bin_edges)
                counts_total, bin_edges_total = np.histogram(df_total["inv-mass"], bins=bin_edges)
                bin_edges = bin_edges_pass
                # Compute bin widths
                bin_widths = np.diff(bin_edges)
                bin_edges = array.array('f', bin_edges)
                hist_pass_pt = ROOT.TH1F("hist", "title", len(bin_edges)-1, bin_edges)
                hist_total_pt = ROOT.TH1F("hist2", "title", len(bin_edges)-1, bin_edges)
                # Fill TH1 bin contents (skip underflow bin 0)
                for i_pass, count in enumerate(counts_pass):
                    hist_pass_pt.SetBinContent(i_pass + 1, count)
                for i_total, count in enumerate(counts_total):
                    hist_total_pt.SetBinContent(i_total + 1, count)

                xmin = 70
                xmax = 110

                if hist_pass_pt.Integral() > 1000:
                    print(hist_pass_pt.Integral())
                    pass_stat = fit_hist(hist_pass_pt, xmin, xmax, os.path.join(plot_dir, f"eta_{eta_min}_ptbin_{i}_{j}_pass"), "run3", 0)
                    total_stat = fit_hist(hist_total_pt, xmin, xmax, os.path.join(plot_dir, f"eta_{eta_min}_ptbin_{i}_{j}_total"), "run3", 0)

                    signal_pass = pass_stat["signal_integral"]
                    signal_total = total_stat["signal_integral"]
                else:
                    signal_pass = df_pass.shape[0]
                    signal_total = df_total.shape[0]

            print(signal_pass, signal_total)

            # Compute bin center and bin number
            lead_center = 0.5 * (leading_pt_low + leading_pt_high)
            sublead_center = 0.5 * (subleading_pt_low + subleading_pt_high)

            bin_idx_pass = hist_pass.FindBin(lead_center, sublead_center)
            bin_idx_total = hist_total.FindBin(lead_center, sublead_center)

#            signal_pass = int(signal_pass)
#            signal_total = int(signal_total)
            # Fill 2D bin content and errors
            hist_pass.SetBinContent(bin_idx_pass, signal_pass)
            hist_pass.SetBinError(bin_idx_pass, np.sqrt(signal_pass))

            hist_total.SetBinContent(bin_idx_total, signal_total)
            hist_total.SetBinError(bin_idx_total, np.sqrt(signal_total))

        eff2D = ROOT.TEfficiency(hist_pass, hist_total)
         # Draw 2D efficiency map
        c = CMS.cmsCanvas(
            f"{wp}_eff2D",
            hist_pass.GetXaxis().GetXmin(),
            hist_pass.GetXaxis().GetXmax(),
            hist_pass.GetYaxis().GetXmin(),
            hist_pass.GetYaxis().GetXmax(),
            "Leading electron p_{T} [GeV]",
            "Subleading electron p_{T} [GeV]",
            square=True,
            extraSpace=0.0,
            iPos=0,
            with_z_axis=True
        )
        # Set text precision to 2 decimal places
        # Get the underlying "efficiency histogram"
        CMS.cmsObjectDraw(eff2D,"COLZ");

        hist = hist_pass
        for i in range(1, hist.GetNbinsX() + 1):
            for j in range(1, hist.GetNbinsY() + 1):
                # Bin center coordinates
                x = hist.GetXaxis().GetBinCenter(i)
                y = hist.GetYaxis().GetBinCenter(j)
                ibin = eff2D.GetGlobalBin(i, j) 
                # Efficiency value
                val = eff2D.GetEfficiency(ibin)
        
                # Asymmetric errors
                err_up = eff2D.GetEfficiencyErrorUp(ibin)
                err_low = eff2D.GetEfficiencyErrorLow(ibin)
        
                # Format: value ± max(error_up, error_low)
                label = ROOT.TLatex()
                label.SetTextAlign(22)  # center
                label.SetTextSize(0.02)
                if val < 0.01:
                  continue
                label.DrawLatex(x, y,  f"{val:.2f}") #^{{+{err_up:.2f}}}_{{-{err_low:.2f}}}")

        # CMS style annotation
        label = ROOT.TLatex()
        label.SetNDC()
        label.SetTextFont(62)
        label.SetTextSize(0.035)
#        label.DrawLatex(0.17, 0.92, f"{wp}, {eta_min} < |#eta| < {eta_max}")

#        DST_text = "Scouting double-EG trigger (2024)" if "DoubleEG" in Target_DST else "Scouting single-Photon trigger (2024)"
#        label.DrawLatex(0.55, 0.92, DST_text)

#        hdf = CMS.GetcmsCanvasHist(c)
#        hdf.GetYaxis().SetLabelSize(0.04)
#        hdf.GetYaxis().SetTitleOffset(1.2)
#        hdf.GetYaxis().SetTitleSize(0.05)
#        hdf.GetXaxis().SetLabelSize(0.04)
#        hdf.GetXaxis().SetTitleOffset(1.2)
#        hdf.GetXaxis().SetTitleSize(0.05)
  
        # Access the histogram drawn on the canvas (or later) and adjust axis titles
        hdf = CMS.GetcmsCanvasHist(c)  # your CMS helper to get the "frame" histogram

        # Make axis titles smaller
        hdf.GetXaxis().SetTitleSize(0.05)  # smaller than default 0.05
        hdf.GetXaxis().SetTitleOffset(1.3) # adjust offset if needed
        hdf.GetYaxis().SetTitleSize(0.05)
        hdf.GetYaxis().SetTitleOffset(1.3)

        c.SetLogx()
        c.SetLogy()


        CMS.SaveCanvas(c, os.path.join(plot_dir, f'{Target_DST}_Efficiency{postfix}.png'), close = False)
        CMS.SaveCanvas(c, os.path.join(plot_dir, f'{Target_DST}_Efficiency{postfix}.pdf'))



# === Run and plot ===
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Explore the structure of an HDF5 file")
    parser.add_argument("--fit", action='store_true')
    parser.add_argument("--store_path", help="Path to the HDF5 file")
    parser.add_argument("--plot_path", help="Path to the HDF5 file")
    # Parse command-line arguments
    args = parser.parse_args()


    # Load the HDF5 file
    plot_dir = args.plot_path
    os.makedirs(plot_dir, exist_ok=True)

    # Match all files like output*.h5 in the given directory
    file_pattern = os.path.join(args.store_path, "output*.h5")
    file_list = glob.glob(file_pattern)

    dfs = []
    for f in file_list:
        try:
            df_part = pd.read_hdf(f)
            dfs.append(df_part)
        except (OSError, KeyError, ValueError) as e:
            print(f"⚠️ Skipping unreadable file: {f} ({e})")

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()  # empty fallback
  
    print(f"✅ Loaded {len(dfs)} valid files out of {len(file_list)} total.")


    # Select entries that pass PFScouting_JetHT
    print("total df", df["inv-mass"].shape[0])
#    df_base = df[(df['PFScouting_JetHT'] == 1.0) | (df['PFScouting_ZeroBias'] == 1.0) | (df["PFScouting_SingleMuon"] == 1.0)]
    #df_base = df[(df['PFScouting_ZeroBias'] == 1.0) | (df["PFScouting_SingleMuon"] == 1.0)]
    df_base = df[(df["PFScouting_SingleMuon"] == 1.0) | (df["PFScouting_DoubleMuon"] == 1.0)]
    df_base = df_base[(df_base["opposite-charge"] > 0)]
    print("after DST", df_base["inv-mass"].shape[0])
    df_base = df_base[(df_base["inv-mass"] < 110) & (df_base["inv-mass"] > 70)]

    Derive_Trigger_Efficiency_orthogonal_2D(
        df_base,
        ((abs(df_base["tag-eta"]) < 1.44) & (df_base["tag-id"] > 3) &(abs(df_base["probe-eta"]) < 1.44) ),
        "",
        "PFScouting_DoubleEG",
        fit = args.fit,
        pt_bins =  np.array([20, 30, 40, 50, 80, 100, 300]),
        postfix = "_2D"
    )


    Derive_Trigger_Efficiency(
        df_base,
        ((df_base["tag-pt"] > df_base["probe-pt"]) & (df_base["tag-id"] > 3) & (df_base["tag-pt"] > 35)),
        "",
        "PFScouting_DoubleEG",
        fit = args.fit,
        pt_bins =  np.array([5, 8, 10, 13, 17, 20, 25, 30, 35, 40, 50, 80, 100, 120, 300]),
        postfix = "_subleading"
    )


    Derive_Trigger_Efficiency(
        df_base,
        ((df_base["tag-pt"] < df_base["probe-pt"]) & (df_base["tag-id"] > 3)),
        "",
        "PFScouting_DoubleEG",
        fit = args.fit,
        pt_bins =  np.array([15, 17, 20, 25, 30, 35, 40, 50, 80, 100, 120, 300]),
        postfix = "_leading"
    )

    Derive_Trigger_Efficiency(
        df_base,
#        ((df_base["tag-pt"] < df_base["probe-pt"]) & (df_base["tag-id"] > 1)),
        ((abs(df_base["tag-eta"]) > 1.57) &  (df_base["tag-id"] > 1) & (abs(df_base["probe-eta"]) < 1.44)),
        "",
        "PFScouting_SinglePhotonEB",
        fit = args.fit,
        pt_bins =  np.array([15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50, 70, 100, 300])
    )
