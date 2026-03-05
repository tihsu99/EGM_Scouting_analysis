import ROOT
import cmsstyle as CMS
import os, sys
from array import array
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
import argparse
import glob


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB values scaled between 0 and 1."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b



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
          cb_amp, cb_alpha, cb_n, cb_mean, cb_sigma):
    
    background = bkg_amp * np.exp(bkg_slope * x)
    cb = crystal_ball(x, cb_alpha, cb_n, cb_mean, cb_sigma, cb_amp)
    return background + cb

  p0 = [1e6, 0.0,        # Background: amp, slope
      2e5, 1.5, 3.0, (xmax + xmin) / 2, 1.0]  # Crystal Ball: amp, alpha, n, mean, sigma

  # --- Fit with scipy ---
  popt, pcov = curve_fit(model, x_vals, y_vals, p0=p0, sigma=y_errs, absolute_sigma=True, maxfev=10000)

  # --- Plot results ---
  x_fit = np.linspace(min(x_vals), max(x_vals), 1000)
  y_fit = model(x_fit, *popt)
 
  fig, ax = plt.subplots()



  ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='ko', label='Data', markersize=3)
  ax.plot(x_fit, y_fit, 'm-', label='Signal + Background')
  ax.plot(x_fit, popt[0] * np.exp(popt[1] * x_fit), 'brown', label='Background')
  ax.plot(x_fit, crystal_ball(x_fit, *popt[3:7], popt[2]), 'cyan', label='Signal(Crystal Ball)')

  # Calculate the background curve
  background_curve = popt[0] * np.exp(popt[1] * x_fit)

  # Calculate the signal curve (Crystal Ball)
  signal_curve = crystal_ball(x_fit, *popt[3:7], popt[2])

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
    "cb_amp", "cb_alpha", "cb_n", "cb_mean", "cb_sigma"
  ]


  print("Fit results:")
  output_dict = dict()
  for name, val in zip(param_names, popt):
    print(f"{name:>10} = {val:.6f}")
    output_dict[name] = val
  return output_dict 


def Draw(data, name, plotdir):

    c = None
    hist_dict = dict()
    hex_colors = [
    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
    "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"
    ]
    default_colors = []

    for i, hex_color in enumerate(hex_colors):
      color_id = ROOT.TColor.GetColor(hex_color)
      default_colors.append(color_id)
    mass_width_result = dict()

    idx = 0
    maximum_y = 0
    for idx_tmp, (data_legend, data_point) in enumerate(data.items()):
     
      if not data_legend.startswith(name):
         continue
      print(data_legend)
      data_legend = data_legend.replace(f"{name}-", "")
      if "Veto" in data_legend:
          continue
      # Pre-binned histogram from NumPy
      # Define bin edges
      edges_lowest = np.arange(0.5, 4, 0.1)
      edges_low = np.arange(4, 15, 0.5)
      edges_high = np.arange(15, 200 + 2, 2)
      bin_edges = np.concatenate([edges_lowest, edges_low[1:], edges_high[1:]])

      # Compute the histogram
      counts, bin_edges = np.histogram(data_point, bins=bin_edges)
      # Compute bin widths
      bin_widths = np.diff(bin_edges)

      # Normalize counts by bin width
      normalized_counts = counts / bin_widths

      hist = ROOT.TH1F("hist", "title", len(bin_edges)-1, bin_edges)
      # Fill TH1 bin contents (skip underflow bin 0)
      for i, count in enumerate(normalized_counts):
         hist.SetBinContent(i + 1, count)

      hist_dict[data_legend] = hist.Clone()
      x_axis = hist_dict[data_legend].GetXaxis()
      y_axis = hist_dict[data_legend].GetYaxis()
      nbinX  = x_axis.GetNbins()
      nbinY  = y_axis.GetNbins()
      x_binnings = [x_axis.GetBinLowEdge(bin_+1) for bin_ in range(nbinX+1)]
      y_binnings = [y_axis.GetBinLowEdge(bin_+1) for bin_ in range(nbinY+1)]


      hist_dict[data_legend].SetDirectory(0)
      fit_results = fit_hist(hist_dict[data_legend], xmin = 2, xmax = 5, fig_name=os.path.join(plotdir, f'{data_legend}_{name}'), era="eraF", lumi = 1)
      mass_width_result[data_legend] = {"mass":  fit_results["cb_mean"], "width": 100 * fit_results["cb_sigma"] / fit_results["cb_mean"]}

#      hist_dict[data_legend] = hist_dict[data_legend].Rebin(2)


   
      if c is None:
        c = CMS.cmsCanvas('', 2, max(bin_edges), 10, hist_dict[data_legend].GetMaximum()*50, 'M_{e,e} [GeV]', 'events/bin width [GeV]', square = CMS.kRectangular, extraSpace=0.01, iPos=11, yTitOffset=0.8)
        legend = CMS.cmsLeg(0.3, 0.8, 0.96, 0.92, textSize=0.033, columns=3)

      CMS.cmsDraw(hist_dict[data_legend], 'HIST', mcolor = default_colors[idx], fstyle = 0, lwidth =2 , lcolor = default_colors[idx], fcolor = 0, msize=0)
      legend.AddEntry(hist_dict[data_legend], f"{data_legend}".replace("iso", "Iso"))
      idx += 1
    c.SetLogx()

#    label = ROOT.TLatex()
#    label.SetNDC()
#    label.SetTextFont(62)  # Bold
#    label.SetTextSize(0.045)
#    label.SetTextAlign(11)  # Left-top corner
#    label.DrawLatex(0.65, 0.85, "PFMonitoring")
#    label.DrawLatex(0.65, 0.8, "dataset")

    c.SetLogy()
#    c.SetLogx()
    hdf = CMS.GetcmsCanvasHist(c)
    hdf.GetYaxis().SetLabelSize(0.025)
    hdf.GetYaxis().SetTitleSize(0.047)
    hdf.GetXaxis().SetLabelSize(0.025)
    hdf.GetXaxis().SetTitleSize(0.047)

    CMS.SaveCanvas(c, os.path.join(plotdir, '{}.png'.format(name)), close = False)
    CMS.SaveCanvas(c, os.path.join(plotdir, '{}.pdf'.format(name)))

    plot_mass_and_width_by_era(mass_width_result, fig_name = os.path.join(plotdir, f'mass_fitting_{name}'))

def plot_mass_and_width_by_era(mass_width_result, fig_name):
    """
    Plots width and mass against era with dual y-axes.

    Parameters:
        mass_width_result (dict): Dictionary with structure:
                                  { 'era': {'mass': float, 'width': float}, ... }
    """
    eras = [era for era in list(mass_width_result.keys()) if "noiso" not in era]
    widths = [mass_width_result[era]['width'] for era in eras if "noiso" not in era]
    masses = [mass_width_result[era]['mass'] for era in eras if "noiso" not in era]
    widths_noiso = [mass_width_result[f"{era}-noiso"]['width'] for era in eras]
    masses_noiso = [mass_width_result[f"{era}-noiso"]['mass'] for era in eras]

    fig, ax1 = plt.subplots()

    color_width = 'tab:blue'
    ax1.set_ylabel('Relative Width [%]', color=color_width)
    ax1.plot(eras, widths, color=color_width, marker='o', label='cutbased-id',  linestyle='-')
    ax1.plot(eras, widths_noiso, color=color_width, marker='o', label='cutbased-id-noIso',  linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_width)
    ax1.set_ylim(3.3, 4.6)
    ax1.grid()
    ax1.legend()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color_mass = 'tab:red'
    ax2.set_ylabel('Mass [GeV]', color=color_mass)
    ax2.plot(eras, masses, color=color_mass, marker='s', label='Mass',  linestyle='-')
    ax2.plot(eras, masses_noiso, color=color_mass, marker='s', label='Mass',  linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_mass)
    ax2.set_ylim(80, 92)
 
    hep.cms.text("Preliminary", loc=2, fontsize=12)
    hep.cms.lumitext(f"2024 eraF (13.6 TeV)")

    fig.tight_layout()
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    plt.show()
    plt.savefig(f"{fig_name}.png")
    plt.savefig(f"{fig_name}.pdf")

# === Run and plot ===
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Explore the structure of an HDF5 file")
    parser.add_argument("--store_path", help="Path to the HDF5 file")
    parser.add_argument("--plot_path", help="Path to the HDF5 file")
    # Parse command-line arguments
    args = parser.parse_args()
    CMS.SetExtraText("Preliminary")
    CMS.SetEnergy("2024 eraF, 13.6")
    CMS.SetLumi("")

    # Directory containing your .npz files
    directory = args.store_path
    pattern = os.path.join(directory, "inv_mass_data*.npz")

    # Find and load the files
    file_list = glob.glob(pattern)
    file_list.sort()  # Optional: ensure consistent order

    # Concatenate a specific array (e.g., 'mass') from all files
    # Load all npz files
    loaded_files = [np.load(f) for f in file_list]

    # Get all keys from the first file
    keys = loaded_files[0].files

    # Concatenate arrays for each key
    data = {key: np.concatenate([f[key] for f in loaded_files]) for key in keys}


    os.makedirs(args.plot_path, exist_ok=True)
    Draw(data, "inv-mass-EBEB", args.plot_path)

    #Draw(data, "inv-mass", args.plot_path)
