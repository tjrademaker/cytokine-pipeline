# -*- coding:utf-8 -*-
"""
Script to import the raw data, smooth it lightly with a moving average,
fit cubic splines to it, and save the cubic splines in a DataFrame.

Search "TODO" to see where to enter the file name (in the main)
and, if desired, the processing parameters (in save_spline_dataframe).


@author:frbourassa
August 13, 2019
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pickle
import os
import seaborn as sns

from cytode.open_dataframes import process_file, log_management_df, read_conc
from cytode.series_backend import nicer_name  # to extract experiment name

# Plot parameters
sns.reset_orig()
sns.set_palette("colorblind", 8)
plt.rcParams["figure.figsize"] = 8, 6
plt.rcParams["lines.linewidth"] = 2.
plt.rcParams["font.size"] = 14.
plt.rcParams["axes.labelsize"] = 16.
plt.rcParams["legend.fontsize"] = 14.
plt.rcParams["xtick.labelsize"] = 14.
plt.rcParams["ytick.labelsize"] = 14.


# For adding subtitles in legends.
# Artist class: handle, containing a string
class LegendSubtitle(object):
    def __init__(self, message, **text_properties):
        self.text = message
        self.text_props = text_properties

    def get_label(self, *args, **kwargs):
        return ""  # no label, the artist itself is the text

# Handler class, give it text properties
class LegendSubtitleHandler(HandlerBase):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle.text, size=fontsize, **orig_handle.text_props)
        handlebox.add_artist(title)
        return title


def find_peptide_concentration_names(index):
    """ Find the level names closest to Peptide and Concentration in the index.
    Warning: some level names are hardcoded here. The list will need to be
    updated if new types of data come in.

    Args:
        (pd.MultiIndex): the index to look in.

    Returns:
        to_remove (list): level name closest to Peptide and level name
            closest to Concentration. Always has length 2.
    """
    # Potential level names equivalent to Peptide and Concentration,
    # in decreasing order of priority
    peptide_lvl_names = ["Peptide", "TumorPeptide", "Agonist", "TCR_Antigen"]
    concentration_lvl_names = ["Concentration", 'IFNgPulseConcentration',
        'TumorCellNumber', 'TCellNumber']
    existing = index.names
    to_remove = []

    for nm in peptide_lvl_names:
        if nm in existing:
            to_remove.append(nm)
            break  # don't look for the less important ones.
    if len(to_remove) == 0:
            to_remove.append("")
    for nm in concentration_lvl_names:
        if nm in existing:
            to_remove.append(nm)
            break
    if len(to_remove) == 1:
        to_remove.append("")

    return to_remove


def index_split(mindex, to_ignore):
    """ Return a list of non-empty labels in the index with the levels
    in to_ignore removed.

    Args:
        mindex (pd.MultiIndex): the DataFrame whose index will be reduced partially.
        to_ignore (list): list of level names to remove.

    Returns:
        index_entries (list): list of labels without the ignored levels.
        not_removed (list): list of levels that could not be found in the index
    """
    level_names = mindex.names
    not_removed = []
    remove_everything = False  # whether something will be left in the index
    for lvl in to_ignore:
        try:
            mindex = mindex.droplevel(lvl)
        except KeyError as e:
            not_removed.append(lvl)
        except ValueError:
            # If all the entries in the index are in to_remove,
            # the last one cannot be removed
            if len(mindex.names) == 1 and mindex.names[0] == lvl:
                remove_everything = True

    if not remove_everything:
        index_entries = list(mindex.unique())
        # If there was only one extra label level, turn manually each
        # label into a list of length one.
        if type(index_entries[0]) != tuple:
            index_entries = [(a,) for a in index_entries]
    else:
        index_entries = []

    return index_entries, not_removed


def save_spline_dataframe(filename):
    """ Import the pickled cytokine data located at filename and
    return and save (in the output/splines folder) a DataFrame containing the
    sp.interpolate.UnivariateSpline objects.

    Args:
        filename (str): path and file name, including extension.

    Returns:
        spline_frame (pd.DataFrame): a DataFrame with the conditions (Peptide,
            Concentration, TCellType, etc.) in the index and "Cytokine"
            in the columns. There is one spline per condition, per cytokine.
        data (pd.DataFrame): the raw data, formatted like the spline_frame,
            but with Time too in the columns (because there is one entry per
            time point here, compared to one spline object covering all times)
        data_log (pd.DataFrame): the raw data in log scale and shifted so
            the minimum value is 1 (chose the minimum value to be 1 to prevent
            having negative values due to cubic splines overshooting)
        data_smooth (pd.DataFrame): the log data smoothed with a moving average
    """
    # TODO: choose processing arguments
    processing_args = dict(
        take_log = True,
        rescale_max = True,
        smooth_size = 3,  # for the moving average
        rtol_splines = 1/5,  # fraction of residuals between raw and moving avg
        keep_mutants = True,
        tcelltype = None, # all types will be loaded if None
        antibody = None,
        genotype = None,
        lod_folder = "data/LOD/"
    )

    # We consider only the 6 most responsive cytokines
    good_cytokines = ["IFNg", "IL-2", "IL-6", "IL-10", "IL-17A", "TNFa"]

    # Import and process the data
    ret = process_file(filename, set(good_cytokines), **processing_args)

    # Return the data at different stages of processing
    # (first 4 returns of process_file)
    spline_frame, data, data_log, data_smooth = ret[0:4]

    # Save the splines to the output/splines/ folder
    prefix = os.path.join("data", "splines", "spline_dataframe_")
    exp_name = nicer_name(filename)
    frame_file_name = prefix + exp_name + ".pkl"
    with open(frame_file_name, "wb") as handle:
        pass#pickle.dump(spline_frame, handle)

    return spline_frame, data, data_log, data_smooth


def color_quality_lightness_quantity(labels, colorcycle=None):
    """A function that returns a list of colors corresponding
    to the series labels. Each peptide gets a different color,
    and each quantity gets a different intensity.

    Args:
        labels (list of tuples): each tuple is (peptide, concentration), with
            the concentration as a string, ending with units of uM, nM, etc.
        colorcycle (list of colors): the colors though which we cycle
            to assign one color to each peptide. If None (default),
            will use the default color cycle in plt.rcParams.
    """
    # If no colorcycle was given, get the default one.
    if colorcycle is None:
        colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Associate a color to each peptide, a lightness to each concentration.
    # zip all peptides together, all conc together.
    pep_list, conc_list = list(zip(*labels))
    # keep one instance of each peptide, in the order they appear
    pep_list = pd.unique(np.array(pep_list))
    conc_list = set(conc_list)
    colors_dict = {}

    for pep in pep_list:
        next_color = colorcycle.pop(0)
        colors_dict[pep] = next_color
        colorcycle.append(next_color)

    # Normalize conc'ns between 0.15 (max) and 0.85 (min) logarithmically
    conc_vals = np.array([read_conc(a) for a in conc_list])
    # Before taking the log, replace 0 by 10 times less the nonzero minconc
    minconc, maxconc = np.amin(conc_vals), np.amax(conc_vals)

    if minconc != maxconc:
        if minconc == 0:
            nonzero_min = np.amin(conc_vals[conc_vals > 0])
            conc_vals[conc_vals == 0] = nonzero_min / 10
            minconc = nonzero_min / 10
        minconc, maxconc = np.log10(minconc), np.log10(maxconc)
        conc_vals = np.log10(conc_vals)
        # Want to map [minconc, maxconc] to [lmin, lmax]
        # with minconc to lmax, maxconc to lmin\
        lmin, lmax = 0.25, 0.85
        def scaler(x):
            return lmax - (lmax - lmin)/(maxconc - minconc) * (x - minconc)
    # Maybe there's only one concentration
    else:
        scaler = lambda x: None  # Use default hsl of that color.

    # To be used as hue, lightness or saturation
    light_dict ={lbl:scaler(conc) for lbl, conc in zip(conc_list, conc_vals)}

    # Finally, build a list of one color per label
    built_colors = []
    for lbl in labels:
        col = colors_dict[lbl[0]]
        brightness = light_dict[lbl[1]]
        # Use the seaborn function to modify the lightness (l) of color col
        built_colors.append(sns.set_hls_values(col, l=brightness))

    return built_colors


def add_processing_legend(ax, handles, example_raw, example_knot=None):
    """ Add a legend in the given ax or fig containing the existing handles (expected
        to be solid lines) and a section explaining the types of curves.
    """
    # Title
    handles.append(LegendSubtitle("Processing:"))

    # Curve shapes
    handles.append(Line2D([0], [0], color='grey',
                            label="Spline", linestyle="-"))
    raw_ls = example_raw.get_linestyle()
    raw_ms = example_raw.get_markersize()
    raw_mew = example_raw.get_markeredgewidth()
    raw_m = example_raw.get_marker()

    handles.append(Line2D([0], [0], marker=raw_m, color='grey',
                             label="Log data", linestyle=raw_ls,
                             markeredgecolor="grey", markersize=raw_ms,
                             markeredgewidth=raw_mew, markerfacecolor="grey"))
    if example_knot is not None:
        knt_m = example_knot.get_marker()
        knt_ls = example_knot.get_linestyle()
        knt_ms = example_knot.get_markersize()
        knt_mew = example_knot.get_markeredgewidth()
        knt_mec = example_knot.get_markeredgecolor()

        handles.append(Line2D([0], [0], marker=knt_m, color='grey',
                            label="Spline knots", linestyle=knt_ls,
                            markeredgecolor=knt_mec, markersize=knt_ms,
                            markeredgewidth=knt_mew, markerfacecolor="grey"))

    ax.legend(handles=handles,
              handler_map={LegendSubtitle: LegendSubtitleHandler()},
              loc="upper left", bbox_to_anchor=(1, 1))


def plot_splines_vs_data(df_spline, df_log, df_smooth, pep_conc_names,
                            spl_times=None, do_knots=False):
    """ For each peptide at all concentrations, plot the cubic splines
    against the rescaled time series of each cytokine. Assumes that
    the index of the DataFrames only contains Peptide and Concentration.

    Args:
        df_spline (pd.DataFrame) : see compare_splines_data's doc
        df_log (pd.DataFrame) : see compare_splines_data's doc
        df_smooth (pd.DataFrame) : see compare_splines_data's doc
        pep_conc_names (list): level names for Peptide and Concentration
        spl_times (1darray): the time axis of the plots
        do_knots (bool): if True, show where the spline knots are.
    Returns:
        fig, axes
    """
    exp_timepoints = df_log.columns.get_level_values("Time").unique()
    if spl_times is None:
        spl_times = np.linspace(0, exp_timepoints[-1], 201)

    # Identify each column to a peptide
    pep_name, conc_name = pep_conc_names
    peptides = df_spline.index.get_level_values(pep_name)
    concentrations = df_spline.index.get_level_values(conc_name)
    column_assignment = {a:i for i, a in enumerate(peptides.unique())}

    # One cytokine per row, one peptide per column
    nrows = len(df_spline.columns.get_level_values("Cytokine").unique())
    ncols = len(peptides.unique())

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey="row")
    fig.set_size_inches(4*(ncols + 1), 4*nrows)

    # Generate colors for peptides and concentrations
    # Assume the index will have the right length
    colors = color_quality_lightness_quantity(list(df_spline.index))
    sr_colors = pd.Series(colors, index=df_spline.index)

    # Plot one cytokine at a time (one row at a time)
    for i, cyt in enumerate(df_spline.columns):
        # Reset the list of curves every cytokine so we have all colors once.
        legend_handles = []
        for j in range(len(peptides)):
            pep, conc = peptides[j], concentrations[j]
            lbl = df_spline.index[j]  # either (pep, conc) or (conc, pep)
            nice_lbl = pep + " [{}]".format(conc)
            col = column_assignment[pep]
            if len(column_assignment) == 1:
                ax=axes[i]
            else:
                ax = axes[i, col]
            clr = sr_colors[lbl]
            spline = df_spline.loc[lbl, cyt]
            smoothcurve = df_smooth.loc[lbl, cyt]
            logcurve = df_log.loc[lbl, cyt]
            # Plot the 2 curves and save the solid curve for the legend
            li, = ax.plot(spl_times, spline(spl_times),
                color=clr, ls="-", label=nice_lbl)
            legend_handles.append(li)
            #ax.plot(exp_timepoints, df_smooth.loc[lbl, cyt], color=clr, ls="--",
                #marker="^", ms=5, mfc=clr, mec=clr)
            # Also plot the knots of that spline.
            rawli, = ax.plot(exp_timepoints, df_log.loc[lbl, cyt], color=clr,
                ls=":", marker="o", ms=5, mfc=clr, mec=clr)
            if do_knots:
                knots = spline.get_knots()
                knotli, = ax.plot(knots, spline(knots),
                    ls="none", marker="^", ms=5, mec="r", mfc=clr, mew=1.)
            else:
                knotli = None
            if i == len(df_spline.columns) - 1:
                ax.set_xlabel("Time [h]")
            elif i == 0:
                ax.set_title(pep, size=16)
            if j == 0:
                ax.set_ylabel("log " + cyt + "[-]")

    # After all the cytokines are done, add a nice categorical legend
    # (peptide, type of curves)
    legax = fig.add_subplot(1, 1, 1)
    legax.set_axis_off()
    add_processing_legend(legax, legend_handles, rawli, knotli)

    return fig, axes


def compare_splines_data(df_spline, df_log, df_smooth, spl_times=None,
                        do_knots=False, showplots=False, nice_name="notitle"):
    """ For each peptide at all concentrations, plot the cubic splines
    against the rescaled time series of each cytokine. Produces one plot
    per TCellType, Genotype, TCellNumber, Antibody.

    Args:
        df_spline (pd.DataFrame): df containing the spline objects.
        df_log (pd.DataFrame): df containing the log+rescaled time series.
        df_smooth (pd.DataFrame): df containing the smoothed time series.
        spl_times (1darray): time axis of the plots.
        do_knots (bool): instead of comparing splines to data, plot
            splines with the position of their knots.
        showplots (bool): if True, each plot will be shown as it is produced.
            False by default.
        nice_name (str): the name of the experiment.

    Returns:
        figures (list of matplotlib.figure.Figure): list of the Figures
        axeslist (list np.2darray): list of arrays of matplotlib.axes.Axes
    """
    # Find the level names for Peptide and Concentration to remove from
    # the indexing of the different plot panels, i.e., the levels in to_remove
    # will show up in each panel of plots.
    to_remove = find_peptide_concentration_names(df_spline.index)
    index_entries, absent = index_split(df_spline.index.copy(), to_remove)

    # If Peptide or Concentration could not be removed, add levels to
    # the DataFrames so after splitting them, there are two levels left
    # to plot each panel
    
    dfs = [df_spline, df_log, df_smooth]
    if len(absent) == 1:
        if absent[0] == to_remove[0]:  # missing Peptide-like entry
            extra = {"Peptide":"Peptide"}
            to_remove[0] = "Peptide"
        elif absent[0] == to_remove[1]:  # missing Concentration-like
            extra = {"Concentration":"Concentration"}
            to_remove[1] = "Concentration"
        for i in range(len(dfs)):
            dfs[i]=dfs[i].assign(**extra).set_index(list(extra.keys()), append=True)

    elif len(absent) == 2:
        lvls = {"Peptide":"Peptide", "Concentration":"Concentration"}
        for i in range(len(dfs)):
            dfs[i].assign(**lvls).set_index(list(extra.keys()), append=True)
        to_remove = ["Peptide", "Concentration"]

    # Reorder the DataFrames to make sure the levels to slice are outer.
    other_levels = list(df_spline.index.names)

    for lvl in to_remove:
        if lvl in other_levels:
            other_levels.remove(lvl)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].reorder_levels(other_levels + to_remove, axis=0)
        #dfs[i] = dfs[i].sort_index()
    df_spline, df_log, df_smooth = dfs

    # If there are multiple TCellType, TCellNumber, etc. in addition to
    # Peptide, Concentration conditions, prepare a different plot for each.
    # Prepare a list of sub-dataframes with only Peptide, Concentration.
    list_spline_df, list_log_df, list_smooth_df = [], [], []
    if len(index_entries) == 0:
        list_spline_df = [df_spline]
        list_log_df = [df_log]
        list_smooth_df = [df_smooth]
        index_entries = [("WT",)]
    else:
        for lbl in index_entries:
            list_spline_df.append(df_spline.loc[lbl])
            list_log_df.append(df_log.loc[lbl])
            list_smooth_df.append(df_smooth.loc[lbl])

    # Prepare one plot per sub-frame. Call a sub-function for this.
    figures = []
    axeslist = []
    for i in range(len(index_entries)):
        dfs =  [list_spline_df[i], list_log_df[i], list_smooth_df[i]]
        # Prepare a title: conditions joined by a comma
        fig_title = nice_name + "_" + "_".join(map(str, index_entries[i]))
        fig_title = fig_title.replace("/", "")  # avoid fake folder changes
        #if do_knots:
            #fig, axes = plot_spline_knots(dfs[0], df_log.columns, spl_times)
        #else:
        print("Drawing a plot panel for", fig_title)
        fig, axes = plot_splines_vs_data(*dfs, to_remove, spl_times, do_knots)
        figures.append(fig)
        axeslist.append(axes)
        if showplots:
            fig.subplots_adjust(top=0.9)  # To prevent title overlap
            fig.suptitle(fig_title, size=18)
            plt.show()
            plt.close()
        else:
            fig.tight_layout()
            figtype = "spline_knots_" if do_knots else "spline_comparison_"
            figpath = os.path.join("output", "plots_splines", nice_name)
            # Create a folder for the dataset
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            fig.savefig(os.path.join(figpath, figtype) + fig_title + ".pdf",
                        format="pdf", dpi=200, transparent=True)
            plt.close()

    return figures, axeslist

# The code for this function is very similar to that of plot_splines_vs_data
def plot_spline_knots(df_spline, df_log_columns, spl_times=None):
    """ For each peptide at all concentrations, plot the cubic splines
    against the rescaled time series of each cytokine. Produces one plot
    per TCellType, Genotype, TCellNumber, Antibody.

    Args:
        df_spline (pd.DataFrame): df containing the spline objects.
        df_log_columns (pd.MultiIndex): columns MultiIndex of the data,
            to extract experimental times.
        spl_times (1darray): the time axis of the plots
    Returns:
        figures (list of matplotlib.figure.Figure): list of the Figures
        axeslist (list np.2darray): list of arrays of matplotlib.axes.Axes
    """
    if spl_times is None:
        timepoints = df_log_columns.get_level_values("Time").unique()
        spl_times = np.linspace(0, timepoints[-1], 201)

    # One cytokine per row, one peptide per column
    nrows = len(df_spline.columns.get_level_values("Cytokine").unique())
    ncols = len(df_spline.index.get_level_values("Peptide").unique())

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey="row")
    fig.set_size_inches(4*(ncols + 1), 4*nrows)
    colors = color_quality_lightness_quantity(list(df_spline.index))
    sr_colors = pd.Series(colors, index=df_spline.index)

    # Identify each column to a peptide
    peptides = df_spline.index.get_level_values("Peptide")
    concentrations = df_spline.index.get_level_values("Concentration")
    column_assignment = {a:i for i, a in enumerate(peptides.unique())}

    # Plot one cytokine at a time (one row at a time)
    for i, cyt in enumerate(df_spline.columns):
        # Reset the list of curves every cytokine so we have all colors once.
        legend_handles = []
        for j in range(len(peptides)):
            pep, conc = peptides[j], concentrations[j]
            lbl = df_spline.index[j]  # either (pep, conc) or (conc, pep)
            nice_lbl = pep + " [{}]".format(conc)
            col = column_assignment[pep]
            ax = axes[i, col]
            clr = sr_colors[lbl]
            spline = df_spline.loc[lbl, cyt]
            # Save the solid curve for the legend
            li, = ax.plot(spl_times, spline(spl_times),
                color=clr, ls="-", label=nice_lbl)
            # Also plot the knots of that spline.
            knots = spline.get_knots()
            ax.plot(knots, spline(knots), ls="none",
                marker="o", ms=8, mec="r", mfc=clr, mew=1.)
            legend_handles.append(li)
            if i == len(df_spline.columns) - 1:
                ax.set_xlabel("Time [h]")
            elif i == 0:
                ax.set_title(pep, size=16)
            if j == 0:
                ax.set_ylabel("log " + cyt + "[-]")

    # After all the cytokines are done, add a nice categorical legend
    # (peptide, type of curves)
    legax = fig.add_subplot(1, 1, 1)
    legax.set_axis_off()
    legax.legend(handles=legend_handles, ncol=2)

    return fig, axes

def knots_info(df_spline):
    """ Generate plots and a DataFrame with information about the cubic splines.
    """
    # Build two DataFrames containing, respectively,
    # the number of knots and the residuals of each spline
    df_knots = pd.DataFrame(np.zeros(df_spline.shape, dtype=int),
        index=df_spline.index, columns=df_spline.columns)
    df_resid = pd.DataFrame(np.zeros(df_spline.shape),
        index=df_spline.index, columns=df_spline.columns)
    for i in range(df_spline.shape[0]):
        for j in range(df_spline.shape[1]):
            df_knots.iat[i, j] = df_spline.iat[i, j].get_knots().size
            df_resid.iat[i, j] = df_spline.iat[i, j].get_residual()
    # Concatenate the two DataFrames into one, with another column level.
    labeled = {"NumberKnots":df_knots, "Residuals":df_resid}
    df_info = pd.concat(labeled, axis=1, names=["SplineInfo"])
    return df_info

def plot_knots_info(df_spline, do_save=False, nice_name=""):
    """ Plot information about the spline knots """
    # If there is more than Peptide, Concentration in the index,
    # prepare one plot per sub-frame.
    df_info = knots_info(df_spline)

    # Find the level names for Peptide and Concentration to remove from
    # the indexing of the different plot panels, i.e., the levels in to_remove
    # will show up in each panel of plots.
    to_remove = find_peptide_concentration_names(df_spline.index)

    # If there are multiple TCellType, TCellNumber, etc. in addition to
    # Peptide, Concentration conditions, prepare a different plot for each.
    # Prepare a list of sub-dataframes with only Peptide, Concentration.
    index_entries, absent = index_split(df_spline.index.copy(), to_remove)

    # If Peptide or Concentration could not be removed, add levels to
    # the DataFrames so after splitting them, there are two levels left
    # to plot each panel
    if len(absent) == 1:
        if absent[0] == to_remove[0]:  # missing Peptide-like entry
            extra = {"Peptide":"Peptide"}
        elif absent[0] == to_remove[1]:  # missing Concentration-like
            extra = {"Concentration":"Concentration"}
        df_info = df_info.assign(**extra).set_index(extra.keys(), append=True)

    elif len(absent) == 2:
        lvls = {"Peptide":"Peptide", "Concentration":"Concentration"}
        df_info = df_info.assign(lvls).set_index(extra.keys(), append=True)

    list_info_df = []
    if len(index_entries) == 0:
        list_info_df = [df_info]
        index_entries = [("WT",)]
    else:
        for lbl in index_entries:
            list_info_df.append(df_info.loc[lbl])

    for i, df in enumerate(list_info_df):
        colors = color_quality_lightness_quantity(list(df.index))
        df = df.stack("Cytokine")
        print(df)

        df["Peptide"] = df.index.get_level_values("Peptide")
        df["Concentration"] = df.index.get_level_values("Concentration")
        df["Cytokine"] = df.index.get_level_values("Cytokine")

        g = sns.relplot(x="Residuals", y="NumberKnots", data=df,
                        hue="Peptide", size="Concentration", col="Cytokine",
                        col_wrap=2, legend="full")
        nrows = len(g.fig.axes) // 2 +len(g.fig.axes) % 2
        g.fig.set_size_inches(2*6, nrows*4)
        g.fig.tight_layout()
        if do_save:
            split_lbl = "_" + "_".join(map(str, index_entries[i]))
            figname = os.path.join("output", "plots", "splines_info_")
            figname = figname + nice_name + split_lbl + ".pdf"
            g.fig.savefig(figname, dpi=200, format="pdf", transparent=True)
        plt.show()
        plt.close()
    return df_info

def main_one_set_from_scratch():
    # TODO: Choose the file to process
    # (path relative to the current working directory or absolute)
    print("Started")
    file_path = "data/data_current/cytokineConcentrationPickleFile-20190725-PeptideTumorComparison_OT1_Timeseries_1-modified.pkl"

    # Generate the splines, save a DataFrame containing cubic splines objects
    df_splines, data, data_log, data_smooth = save_spline_dataframe(file_path)

    # Plot the splines against the data.
    #compare_splines_data(df_splines, data_log, data_smooth, showplots=False,
                #do_knots=False, nice_name=nicer_name(file_path))

    # Put the spline knots as well on the plot against the data
    # To have only splines and their knots, go uncomment the usage of
    # plot_spline_knots in compare_splines_data.
    compare_splines_data(df_splines, data_log, data_smooth, showplots=True,
                do_knots=True, nice_name=nicer_name(file_path))

    df = knots_info(df_splines)
    print(df["NumberKnots"].mean(axis=1))
    print(df["Residuals"].mean(axis=1))
    plot_knots_info(df_splines, do_save=False, nice_name=nicer_name(file_path))

def main_process_all_data():
    # Processing arguments used by Thomas
    processing_args = {
        "take_log": True,   # Use the log of the cytokines if True
        "rescale_max" : False,  # Rescale by the max of the cytokines
        "smooth_size": 3,   # For the moving average, odd number, minimum 3
        "genotype": None,   # Genotype to keep; all mutants kept if None
        "tcelltype": None,  # TCellType to keep; all kept if None
        "antibody": None,   # Antibody condition to keep, all kept if None
        "rtol_splines": 1/2, # Tolerance on the splines
        "lod_folder": "data/LOD/"  # Folder containing LOD files.
    }
    cyto_keep = {"IFNg", "IL-2", "IL-6", "IL-10", "IL-17A", "TNFa"}
    # Loop over data files in the specified CytoSeries folder
    # folder_data = "data/data_current/"
    folder_data = "data/data_run/"
    
    available_data = [fi for fi in os.listdir(folder_data) if fi.endswith(".pkl")]
    for fi in available_data:
        nicename = nicer_name(fi)
        print("Starting to process and plot", nicename)
        res = process_file(folder_data + fi, cyto_keep, **processing_args)
        [proc_inpt, raw0, raw, smooth, cyto_lbl, series_labels,
                    level_labels, exp_times] = res
        # Define the time interval over which to plot the splines
        spl_times = np.linspace(0, exp_times[-1], 201)
        # Plot and save the splines
        compare_splines_data(proc_inpt, raw, smooth, spl_times,
                    showplots=False, do_knots=False, nice_name=nicename)


if __name__ == "__main__":
    #main_one_set_from_scratch()
    main_process_all_data()
