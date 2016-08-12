# -*- coding: utf-8 -*-

"""
@package pycl
@copyright  [GNU General Public License v2](http://www.gnu.org/licenses/gpl-2.0.html)
@author     Adrien Leger - 2016
* <aleg@ebi.ac.uk>
* [Github](https://github.com/a-slide)
"""

# Strandard library imports

# Third party packages
import numpy as np
import pylab as pl
import pandas as pd
#import seaborn as sns
import matplotlib as mpl

#~~~~~~~ RNASeq plots ~~~~~~~#

def volcano_plot (
    df, X, Y,
    FDR=0.05,
    X_cutoff = 1,
    sig_color="0.40",
    non_sig_color="0.70",
    highlight_list=[],
    **kwargs
    ):
    """
    Run a command line in the default shell and return the standard output
    @param  df  Panda dataframe containing the results. Each line corresponds to a single gene/transcript value. Gene/transcript are
                identified by a target_id column. The other covariate columns need to contain the values for X and Y plotting  
    @param  X   Name of the column for X plotting (usually log2FC)
    @param  Y   Name of the column for Y plotting (usually pvalue)
    @param  FDR false discovery rate cut-off for the Y axis (on the raw value before log transformation for plotting [DEFAULT: 0.05]
    @param  X_cutoff    value for significance cut-off for the X axis [DEFAULT: 1]
    @param  sig_color   Color of the significant points [DEFAULT: "0.40"] 
    @param  non_sig_color Color of the non-significant points [DEFAULT: "0.70"]
    @param  highlight_list  List of dictionaries for values to highlight. Each entry contains:
                [mandatory]     "target_id": List of target_id matching target_id s in df
                [facultative]   "color": A valid matplotlib color, else a random color will be attributed
                [facultative]   "label": A str label, else no label will be given to the serie.
                example: highlight_list= [
                    {"target_id":["id1","id3"], "color":"red", "label":"s1"}
                    {"target_id":["id4","id7","id9"], "color":"green", "label":"s2"}]
    @param  kwargs  Additional parameters for plot appearance derived from pylab basic plot arguments such as:
                figsize, xlim, ylim, title, xlabel, ylabel, bg_color, grid_color, marker, alpha...
    """
        
    # Define default figure parameters
    _plot_preprocessing(kwargs)

    # x and y limits need to be defined now for several options
    if "xlim" not in kwargs:
        kwargs["xlim"] = [df[X].min()-1, df[X].max()+1]
    xlim=kwargs["xlim"]
    if "ylim" not in kwargs:
        kwargs["ylim"] = [-np.log10(df[Y].max())-1, -np.log10(df[Y].min())+1]
    ylim=kwargs["ylim"]
    
    # Extract the significant values
    sig_df = df.query("{0}<={1} and ({2}<=-{3} or {2} >= {3})".format(Y, FDR, X, X_cutoff))
    
    # Plot non significant and significant values
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1
    marker = kwargs["marker"] if "marker" in kwargs else "o"
    linewidth = kwargs["linewidth"] if "linewidth" in kwargs else 0
    pl.scatter(df[X], -np.log10(df[Y]), color=non_sig_color, label='Non significant', marker=marker, linewidth=linewidth, alpha=alpha)
    pl.scatter(sig_df[X], -np.log10(sig_df[Y]), color=sig_color, label='Significant', marker=marker, linewidth=linewidth, alpha=alpha)
    
    # Add the localization for the different categories
    for val in highlight_list:
        
        # Extract of define default values = Can be expanded later
        target_id_list = val["target_id"] if "target_id" in val else []
        color = val["color"] if "color" in val else np.random.rand(3,1)
        label = val["label"] if "label" in val else ""
        alpha = val["alpha"] if "alpha" in val else 1
        marker = val["marker"] if "marker" in val else 'o'
        linewidth = val["linewidth"] if "linewidth" in val else 0
        
        # Plot the additonal series
        val_df = df[(df.target_id.isin(target_id_list))]
        pl.scatter(val_df[X], -np.log10(val_df[Y]), c=color, label=label, marker=marker, linewidth=linewidth, alpha=alpha)

    # Ploting shaping lines and significance lines
    pl.hlines(0, xlim[0], xlim[1], colors='0.4', linestyles='--', linewidth=2, alpha=0.5)
    pl.vlines(0, ylim[0], ylim[1], colors='0.4', linestyles='--', linewidth=2, alpha=0.5)
    pl.hlines(-np.log10(FDR), xlim[0], xlim[1], colors='0.6', linestyles=':', linewidth=2, alpha=0.5)
    pl.vlines(-X_cutoff, ylim[0], ylim[1], colors='0.6', linestyles=':', linewidth=2, alpha=0.5)
    pl.vlines(X_cutoff, ylim[0], ylim[1], colors='0.6', linestyles=':', linewidth=2, alpha=0.5)
    
    # Tweak the graph
    kwargs["title"]="Volcano Plot  FDR={} Xcutoff={}  Targets={}  Significant Targets={}".format(
        FDR, X_cutoff, len(df.target_id), len(sig_df.target_id))
    _plot_postprocessing(kwargs)

def MA_plot (
    df, X, Y,
    FDR=0.05,
    FDR_col="pval",
    sig_color="0.40",
    non_sig_color="0.70",
    highlight_list=[],
    **kwargs
    ):
    """
    Run a command line in the default shell and return the standard output
    @param  df  Panda dataframe containing the results. Each line corresponds to a single gene/transcript value. Gene/transcript are
                identified by a target_id column. The other covariate columns need to contain the values for X and Y plotting  
    @param  X   Name of the column for X plotting (usually Mean expression)
    @param  Y   Name of the column for Y plotting (usually log2FC)
    @param  FDR false discovery rate cut-off for the Y axis (on the raw value before log transformation for plotting [DEFAULT: 0.05]
    @param  FDR_col Name of the column to use to determine the significance cut-off (usually pvalue)
    @param  sig_color   Color of the significant points [DEFAULT: "0.40"] 
    @param  non_sig_color Color of the non-significant points [DEFAULT: "0.70"] 
    @param  highlight_list  List of dictionaries for values to highlight. Each entry contains:
                [mandatory]     "target_id": List of target_id matching target_id s in df
                [facultative]   "color": A valid matplotlib color, else a random color will be attributed
                [facultative]   "label": A str label, else no label will be given to the serie.
                example: highlight_list= [
                    {"target_id":["id1","id3"], "color":"red", "label":"s1"}
                    {"target_id":["id4","id7","id9"], "color":"green", "label":"s2"}]
    @param  kwargs  Additional parameters for plot appearance derived from pylab basic plot arguments such as:
                figsize, xlim, ylim, title, xlabel, ylabel, bg_color, grid_color...
    """
        
    # Define default figure parameters
    _plot_preprocessing(kwargs)
           
    # x and y limits need to be defined now for several options
    if "xlim" not in kwargs:
        kwargs["xlim"] = [df[X].min()-1, df[X].max()+1]
    xlim=kwargs["xlim"]
    if "ylim" not in kwargs:
        kwargs["ylim"] = [df[Y].min()-1, df[Y].max()+1]
    ylim=kwargs["ylim"]

    # Extract the significant values
    sig_df = df.query("{}<={}".format(FDR_col, FDR))
    
    # Plot non significant and significant values
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1
    marker = kwargs["marker"] if "marker" in kwargs else "o"
    linewidth = kwargs["linewidth"] if "linewidth" in kwargs else 0
    pl.scatter(df[X], df[Y], color=non_sig_color, label='Non significant', marker=marker, linewidth=linewidth, alpha=alpha)
    pl.scatter(sig_df[X], sig_df[Y], color=sig_color, label='Significant', marker=marker, linewidth=linewidth, alpha=alpha)
    
    # Add the localization for the different categories
    for val in highlight_list:
        
        # Extract of define default values = Can be expanded later
        target_id_list = val["target_id"] if "target_id" in val else []
        color = val["color"] if "color" in val else np.random.rand(3,1)
        label = val["label"] if "label" in val else ""
        alpha = val["alpha"] if "alpha" in val else 1
        marker = val["marker"] if "marker" in val else 'o'
        linewidth = val["linewidth"] if "linewidth" in val else 0
        
        # Plot the additonal series
        val_df = df[(df.target_id.isin(target_id_list))]
        pl.scatter(val_df[X], val_df[Y], c=color, label=label, marker=marker, linewidth=linewidth, alpha=alpha)

    # Ploting shaping lines 
    pl.hlines(0, xlim[0], xlim[1], colors='0.4', linestyles='--', linewidth=2, alpha=0.5)

    # Tweak the graph
    kwargs["title"]="MA Plot  FDR={} Targets={}  Significant Targets={}".format(FDR, len(df.target_id), len(sig_df.target_id))
    _plot_postprocessing(kwargs)

def Density_plot (
    df, X, Y,
    FDR=0.05,
    FDR_col="pval",
    sig_color="0.40",
    non_sig_color="0.70",
    highlight_list=[],
    **kwargs
    ):
    """
    Run a command line in the default shell and return the standard output
    @param  df  Panda dataframe containing the results. Each line corresponds to a single gene/transcript value. Gene/transcript are
                identified by a target_id column. The other covariate columns need to contain the values for X and Y plotting  
    @param  X   Name of the column for X plotting (usually Mean expression)
    @param  Y   Name of the column for Y plotting (usually log2FC)
    @param  FDR false discovery rate cut-off for the Y axis (on the raw value before log transformation for plotting [DEFAULT: 0.05]
    @param  FDR_col Name of the column to use to determine the significance cut-off (usually pvalue)
    @param  sig_color   Color of the significant points [DEFAULT: "0.40"] 
    @param  non_sig_color Color of the non-significant points [DEFAULT: "0.70"] 
    @param  highlight_list  List of dictionaries for values to highlight. Each entry contains:
                [mandatory]     "target_id": List of target_id matching target_id s in df
                [facultative]   "color": A valid matplotlib color, else a random color will be attributed
                [facultative]   "label": A str label, else no label will be given to the serie.
                example: highlight_list= [
                    {"target_id":["id1","id3"], "color":"red", "label":"s1"}
                    {"target_id":["id4","id7","id9"], "color":"green", "label":"s2"}]
    @param  kwargs  Additional parameters for plot appearance derived from pylab basic plot arguments such as:
                figsize, xlim, ylim, title, xlabel, ylabel, bg_color, grid_color...
    """
        
    # Define default figure parameters
    _plot_preprocessing(kwargs)
           
    # x and y limits need to be defined now for several options
    if "xlim" not in kwargs:
        kwargs["xlim"] = [df[X].min()-1, df[X].max()+1]
    xlim=kwargs["xlim"]
    if "ylim" not in kwargs:
        kwargs["ylim"] = [df[Y].min()-1, df[Y].max()+1]
    ylim=kwargs["ylim"]

    # Extract the significant values
    sig_df = df.query("{}<={}".format(FDR_col, FDR))
    
    # Plot non significant and significant values
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1
    marker = kwargs["marker"] if "marker" in kwargs else "o"
    linewidth = kwargs["linewidth"] if "linewidth" in kwargs else 0
    pl.scatter(df[X], df[Y], color=non_sig_color, label='Non significant', marker=marker, linewidth=linewidth, alpha=alpha)
    pl.scatter(sig_df[X], sig_df[Y], color=sig_color, label='Significant', marker=marker, linewidth=linewidth, alpha=alpha)
    
    # Add the localization for the different categories
    for val in highlight_list:
        
        # Extract of define default values = Can be expanded later
        target_id_list = val["target_id"] if "target_id" in val else []
        color = val["color"] if "color" in val else np.random.rand(3,1)
        label = val["label"] if "label" in val else ""
        marker = val["marker"] if "marker" in val else 'o'
        linewidth = val["linewidth"] if "linewidth" in val else 0
        
        # Plot the additonal series
        val_df = df[(df.target_id.isin(target_id_list))]
        pl.scatter(val_df[X], val_df[Y], c=color, label=label, marker=marker, linewidth=linewidth, alpha=alpha)

    # Ploting shaping lines 
    pl.hlines(0, xlim[0], xlim[1], colors='0.4', linestyles='--', linewidth=2, alpha=0.5)

    # Tweak the graph
    kwargs["title"]="MA Plot  FDR={} Targets={}  Significant Targets={}".format(FDR, len(df.target_id), len(sig_df.target_id))
    _plot_postprocessing(kwargs)


#~~~~~~~ Generic utilities ~~~~~~~#

def get_color_list(n, colormap="brg", plot_palette=False):
    """
    Return a list of l length with gradient colors from a given matplot lib colormap palette
    @param n    Number of color scalar in the list
    @param  colormap    colormap color palette from matplotlib package see http://matplotlib.org/examples/color/colormaps_reference.html
                        exemple : inferno magma hot blues cool spring winter brg ocean hsv jet ... [DEFAULT: brg]
    @param  plot_palette    if True will plot the palette for visualization purpose [DEFAULT: False]
    @return A list of color codes that can be used for plotting
    """
    
    # Prepare the figure if required
    if plot_palette:
        pl.figure(figsize=(n/2,1))
        pl.xlim(-1, n+1)
        pl.axis("off")
    
    # Init variables
    cmap = mpl.cm.get_cmap(colormap)
    l = []
    index = 0
    step = cmap.N/n
    
    # Icreate the list of colors
    for i in range (n):
        color = cmap(int(index))
        l.append(color)
        index+=step
        
        # Plot the figure if required
        if plot_palette:
            pl.scatter(i, 0, c=color, s=500, linewidth=0)
    
    return l

def plot_text (text, plot_len=20, align="center", **kwargs):
    """
    Plot a text alone as graph. Usefull to separate series of data plots in interactive session.
    @param text Test message to plot
    @param len_plot Length of the plotting area [DEFAULT: 20]
    @param align    Alignment of the text ['left' | 'right' | 'center' ] [DEFAULT: 'center']
    @param kwargs   Additional parameters from matplotlib.text.Text class see http://matplotlib.org/users/text_intro.html
                    Examples = color, family, fontname, position...
                    
    """
    # Define default figure parameters
    p = pl.figure(figsize=[plot_len,0.5],frameon=False)
    p = pl.axis("off")
    
    # Deal with the alignment of the text
    if align == "left":
        x=0
    elif align == "right":
        x=1
    else:
        x=0.5
    kwargs["horizontalalignment"]=align
    
    # Plot the text
    p = pl.text(x, 0.5, text, kwargs)

#~~~~~~~ Private helper methods ~~~~~~~#

def _plot_preprocessing(kws):
    pl.figure(
        figsize=(kws["figsize"] if "figsize" in kws else None),
        frameon=False)
    pl.axes(
        axisbg=(kws["bg_color"] if "bg_color" in kws else "white"),
        frameon=False)
    pl.grid(
        color=(kws["grid_color"] if "grid_color" in kws else "0.9"),
        linestyle='-',
        linewidth=2,
        alpha=0.25)
    pl.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False)

def _plot_postprocessing(kws):
    if "xlabel" in kws:
        pl.xlabel(kws["xlabel"])
    if "ylabel" in kws:
        pl.ylabel(kws["ylabel"])
    if "xlim" in kws:
        pl.xlim(kws["xlim"])
    if "ylim" in kws:
        pl.ylim(kws["ylim"])
    pl.legend(
        bbox_to_anchor=(1, 1),
        loc=2,
        frameon=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
