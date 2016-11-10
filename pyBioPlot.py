# -*- coding: utf-8 -*-

"""
@package pyBioPlot
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
import seaborn as sns
import matplotlib as mpl

#~~~~~~~ RNASeq plots ~~~~~~~#

def volcano_plot (
    df, X, Y,
    FDR=1,
    X_cutoff = 1,
    sig_pos_color="0.5",
    sig_neg_color="0.7",
    non_sig_color="0.9",
    highlight_list=[],
    highlight_palette="Set1",
    highlight_FDR=None,
    highlight_min_targets=0,
    **kwargs
    ):
    """
    Plot a volcano plot with matplot lib from a dataframe in “stacked” or “record” format
    *  df
        Panda dataframe containing the results. Each line corresponds to a single gene/transcript value. Gene/transcript are
        identified by a target_id column. The other covariate columns need to contain the values for X and Y plotting  
    *  X
        Name of the column for X plotting (usually log2FC)
    *  Y
        Name of the column for Y plotting (usually pvalue)
    *  FDR
        false discovery rate cut-off for the Y axis (on the raw value before log transformation for plotting [DEFAULT: 1]
    *  X_cutoff
        value for significance cut-off for the X axis [DEFAULT: 1]
    *  sig_pos_color
        Color of the significant points on the positive side of the X axis [DEFAULT: "0.5"]
    *  sig_neg_color
        Color of the significant points on the negative side of the Y axis [DEFAULT: "0.7"]
    *  non_sig_color
        Color of the non-significant points [DEFAULT: "0.9"]
    *  highlight_list
        List of dictionaries for values to highlight. Each entry contains:
                [mandatory]     "target_id": List or pandas series of target_id matching target_id in the main df
            OR  [mandatory]     "df": A dataframe containing values with X and Y columns can be provided instead of the target_id list
                [facultative]   "color": A valid matplotlib color, else a random color will be attributed
                [facultative]   "label": A str label [DEFAULT: # of the series]
                [facultative]   "alpha": Alpha parameter [DEFAULT: 1]
                [facultative]   "marker": A matplotlib symbol [DEFAULT: "o"]
                example: highlight_list= [
                    {"target_id":["id1","id3"], "color":"red", "label":"s1"}
                    {"target_id":["id4","id7","id9"], "color":"green", "label":"s2", "marker":">", "alpha":0.5}]
    *  highlight_palette
        palette to be used to automatically assign colors to each element of the highlight_list [DEFAULT: "Set1"]
    *  highlight_FDR
        if a value if given the highlight list will be thresholded by the given FDR [DEFAULT: None]
    *  highlight_min_targets
        Mininal number of targets in a series of the highlight list. If below the series will not be plotted  
        [DEFAULT: 0]
    *  kwargs
        Additional parameters for plot appearance derived from pylab basic plot arguments such as:
        figsize, xlim, ylim, title, xlabel, ylabel, bg_color, grid_color, marker, alpha...
    """
    
    # Default values for this plot
    default_val = {
        "linewidth":0,
        "marker":"o",
        "alpha":0.5,
        "xlim":[df[X].min()-1, df[X].max()+1],
        "ylim":[-np.log10(df[Y].max())-1, -np.log10(df[Y].min())+1]}
        
    # Define default figure parameters
    _plot_preprocessing(kwargs)
           
    # x and y limits need to be defined now for several options
    kwargs["xlim"] = kwargs.get("xlim", default_val["xlim"])
    kwargs["ylim"] = kwargs.get("ylim", default_val["ylim"])
    
    # Plot non significant values   
    df = df.dropna()   
    pl.scatter(
        df[X], -np.log10(df[Y]), color=non_sig_color,
        label='Non significant  n={}'.format(len(df)),
        linewidth=kwargs.get("linewidth", default_val["linewidth"]),
        marker=kwargs.get("marker", default_val["marker"]),
        alpha=kwargs.get("alpha", default_val["alpha"]))
        
    # Plot significant positive values
    sig_df = df[(df[Y]<=FDR) & (df[X]>0)]
    pl.scatter( sig_df[X], -np.log10(sig_df[Y]), color=sig_pos_color, label='Significant positive n={}'.format(len(sig_df)),
        linewidth=kwargs.get("linewidth", default_val["linewidth"]),
        marker=kwargs.get("marker", default_val["marker"]),
        alpha=kwargs.get("alpha", default_val["alpha"]))

    # Plot significant negative values
    sig_df = df[(df[Y]<=FDR) & (df[X]<0)]
    pl.scatter( sig_df[X], -np.log10(sig_df[Y]), color=sig_neg_color, label='Significant negative n={}'.format(len(sig_df)),
        linewidth=kwargs.get("linewidth", default_val["linewidth"]),
        marker=kwargs.get("marker", default_val["marker"]),
        alpha=kwargs.get("alpha", default_val["alpha"]))
    
    # Highlight the categories given in the highlight list
    highlight_list = _parse_highlight_list (
        highlight_list=highlight_list,
        df=df,
        default_val=default_val,
        highlight_palette=highlight_palette,
        FDR=highlight_FDR,
        FDR_col=Y,
        min_targets=highlight_min_targets)
        
    for h in highlight_list:
        # Plot the additional series
        pl.scatter(
            h["df"][X],
            -np.log10(h["df"][Y]),
            color=h["color"],
            label=h["label"],
            marker=h["marker"],
            linewidth=h["linewidth"],
            alpha=h["alpha"])

    # Ploting shaping lines and significance lines
    pl.hlines(0, kwargs["xlim"][0], kwargs["xlim"][1], colors='0.4', linestyles='--', linewidth=2, alpha=0.5)
    pl.vlines(0, kwargs["ylim"][0], kwargs["ylim"][1], colors='0.4', linestyles='--', linewidth=2, alpha=0.5)
    pl.hlines(-np.log10(FDR), kwargs["xlim"][0], kwargs["xlim"][1], colors='0.6', linestyles=':', linewidth=2, alpha=0.5)
    
    # Tweak the graph
    kwargs["title"]="Volcano Plot  FDR={}".format(FDR)
    _plot_postprocessing(kwargs)

def MA_plot (
    df, X, Y,
    FDR=1,
    FDR_col="pval",
    sig_pos_color="0.5",
    sig_neg_color="0.7",
    non_sig_color="0.9",
    highlight_list=[],
    highlight_palette="Set1",
    highlight_FDR=None,
    highlight_min_targets=0,
    **kwargs
    ):
    """
    Plot a MA plot with matplotlib from a dataframe in “stacked” or “record” format
    *  df
        Panda dataframe containing the results. Each line corresponds to a single gene/transcript value. Gene/transcript are
        identified by a target_id column. The other covariate columns need to contain the values for X and Y plotting  
    *  X
        Name of the column for X plotting (usually Mean expression)
    *  Y
        Name of the column for Y plotting (usually log2FC)
    *  FDR
        False discovery rate cut-off for the Y axis (on the raw value before log transformation for plotting [DEFAULT: 1]
    *  FDR_col
        Name of the column to use to determine the significance cut-off (usually pvalue)
    *  sig_pos_color
        Color of the significant points on the positive side of the X axis [DEFAULT: "0.5"]
    *  sig_neg_color
        Color of the significant points on the negative side of the Y axis [DEFAULT: "0.7"]
    *  non_sig_color
        Color of the non-significant points [DEFAULT: "0.9"]
    *  highlight_list
        List of dictionaries for values to highlight. Each entry contains:
                [mandatory]     "target_id": List or pandas series of target_id matching target_id in the main df
            OR  [mandatory]     "df": A dataframe containing values with X and Y columns can be provided instead of the target_id list
                [facultative]   "color": A valid matplotlib color, else a random color will be attributed
                [facultative]   "label": A str label [DEFAULT: # of the series]
                [facultative]   "alpha": Alpha parameter [DEFAULT: 1]
                [facultative]   "marker": A matplotlib symbol [DEFAULT: "o"]
                example: highlight_list= [
                    {"target_id":["id1","id3"], "color":"red", "label":"s1"}
                    {"target_id":["id4","id7","id9"], "color":"green", "label":"s2", "marker":">", "alpha":0.5}]
    *  highlight_palette
        palette to be used to automatically assign colors to each element of the highlight_list [DEFAULT: "Set1"]
    *  highlight_FDR
        if a value if given the highlight list will be thresholded by the given FDR [DEFAULT: None]
    *  highlight_min_targets
        Mininal number of targets in a series of the highlight list. If below the series will not be plotted  
        [DEFAULT: 0]
    *  kwargs
        Additional parameters for plot appearance derived from pylab basic plot arguments such as:
        figsize, xlim, ylim, title, xlabel, ylabel, bg_color, grid_color...
    """
        
    # Default values for this plot
    default_val = {
        "linewidth":0,
        "marker":"o",
        "alpha":0.5,
        "xlim":[df[X].min()-1, df[X].max()+1],
        "ylim":[df[Y].min()-1, df[Y].max()+1]}
        
    # Define default figure parameters
    _plot_preprocessing(kwargs)
           
    # x and y limits need to be defined now for several options
    kwargs["xlim"] = kwargs.get("xlim", default_val["xlim"])
    kwargs["ylim"] = kwargs.get("ylim", default_val["ylim"])
    
    # Plot non significant values   
    df = df.dropna()   
    pl.scatter(
        df[X], df[Y], color=non_sig_color,
        label='Non significant  n={}'.format(len(df)),
        linewidth=kwargs.get("linewidth", default_val["linewidth"]),
        marker=kwargs.get("marker", default_val["marker"]),
        alpha=kwargs.get("alpha", default_val["alpha"]))
        
    # Plot significant positive values
    sig_df = df[(df[FDR_col]<=FDR) & (df[Y]>0)]
    pl.scatter( sig_df[X], sig_df[Y], color=sig_pos_color, label='Significant positive n={}'.format(len(sig_df)),
        linewidth=kwargs.get("linewidth", default_val["linewidth"]),
        marker=kwargs.get("marker", default_val["marker"]),
        alpha=kwargs.get("alpha", default_val["alpha"]))

    # Plot significant negative values
    sig_df = df[(df[FDR_col]<=FDR) & (df[Y]<0)]
    pl.scatter( sig_df[X], sig_df[Y], color=sig_neg_color, label='Significant negative n={}'.format(len(sig_df)),
        linewidth=kwargs.get("linewidth", default_val["linewidth"]),
        marker=kwargs.get("marker", default_val["marker"]),
        alpha=kwargs.get("alpha", default_val["alpha"]))
        
    # Highlight the categories given in the highlight list
    highlight_list = _parse_highlight_list (
        highlight_list=highlight_list,
        df=df,
        default_val=default_val,
        highlight_palette=highlight_palette,
        FDR=highlight_FDR,
        FDR_col=FDR_col,
        min_targets=highlight_min_targets)
    
    for h in highlight_list:
        # Plot the additional series
        pl.scatter(
            h["df"][X],
            h["df"][Y],
            color=h["color"],
            label=h["label"],
            marker=h["marker"],
            linewidth=h["linewidth"],
            alpha=h["alpha"])

    # Ploting shaping lines 
    pl.hlines(0, kwargs["xlim"][0], kwargs["xlim"][1], colors='0.4', linestyles='--', linewidth=2, alpha=0.5)

    # Tweak the graph
    kwargs["title"]="MA Plot  FDR={}".format(FDR)
    _plot_postprocessing(kwargs)

def density_plot (
    df, X,
    FDR=None,
    FDR_col="pval",
    cumulative=False,
    cut=3,
    highlight_list=[],
    highlight_palette="Set1",
    highlight_FDR=None,
    highlight_min_targets=0,
    **kwargs
    ):
    """
    Plot a density with matplotlib from a dataframe in “stacked” or “record” format
    *  df
        Panda dataframe containing the results. Each line corresponds to a single gene/transcript value. Gene/transcript are
        identified by a target_id column. The other covariate columns need to contain the values for X and Y plotting  
    *  X
        Name of the column to calculate density (usually Mean expression)
    *  FDR
        false discovery rate cut-off for the Y axis (on the raw value before log transformation for plotting [DEFAULT: None]
    *  FDR_col
        Name of the column to use to determine the significance cut-off (usually pvalue)
    *  cumulative
        If true, will plot a cumulative distribution [DEFAULT: 1]
    *  highlight_list
        List of dictionaries for values to highlight. Each entry contains:
                [mandatory]     "target_id": List or pandas series of target_id matching target_id in the main df
            OR  [mandatory]     "df": A dataframe containing values with X and Y columns can be provided instead of the target_id list
                [facultative]   "color": A valid matplotlib color, else a random color will be attributed
                [facultative]   "label": A str label [DEFAULT: # of the series]
                [facultative]   "alpha": Alpha parameter [DEFAULT: 1]
                [facultative]   "linestyle":A matplotlib linestyle [DEFAULT: "-"]
                [facultative]   "linestyle": A matplotlib linestyle [DEFAULT: "-"]
                [facultative]   "linewidth": Width of the line [DEFAULT: 2]
                example: highlight_list= [
                    {"target_id":["id1","id3"], "color":"red", "label":"s1"}
                    {"target_id":["id4","id7","id9"], "color":"green", "label":"s2", "marker":">", "alpha":0.5}]
    *  highlight_palette
        palette to be used to automatically assign colors to each element of the highlight_list [DEFAULT: "Set1"]
    *  highlight_FDR
        if a value if given the highlight list will be thresholded by the given FDR [DEFAULT: None]
    *  highlight_min_targets
        Mininal number of targets in a series of the highlight list. If below the series will not be plotted  
        [DEFAULT: 0]
    *  kwargs
        Additional parameters for plot appearance derived from pylab basic plot arguments such as:
        figsize, xlim, ylim, title, xlabel, ylabel, bg_color, grid_color...
    """
    
    # Default values for this plot
    default_val = {
        "color":"black",
        "linewidth":2,
        "linestyle":"-",
        "alpha":1,
        "xlim":[df[X].min()-1, df[X].max()+1],
        "ylim":[-0.05, 1.05]}
        
    # Define default figure parameters
    _plot_preprocessing(kwargs)
           
    # x and y limits need to be defined now for several options
    kwargs["xlim"] = kwargs.get("xlim", default_val["xlim"])
    kwargs["ylim"] = kwargs.get("ylim", default_val["ylim"])

    # Remove empty values, thresholf with FDR and plot all the values from the df   
    df = df.dropna()
    if FDR:
        df = df[(df[FDR_col]<=FDR)]
    
    sns.kdeplot(
        df[X],
        cumulative=cumulative,
        cut=cut,
        label='All n={}'.format(len(df)),
        color=kwargs.get("color", default_val["color"]),
        linewidth=kwargs.get("linewidth", default_val["linewidth"]),
        linestyle=kwargs.get("linestyle", default_val["linestyle"]),
        alpha=kwargs.get("alpha", default_val["alpha"]))

    # Highlight the categories given in the highlight list
    highlight_list = _parse_highlight_list (
        highlight_list=highlight_list,
        df=df,
        default_val=default_val,
        highlight_palette=highlight_palette,
        FDR=highlight_FDR,
        FDR_col=FDR_col,
        min_targets=highlight_min_targets)
        
    for h in highlight_list:
        # Plot the additional series
        sns.kdeplot(
            h["df"][X],
            cumulative=cumulative,
            cut=cut,
            label=h["label"],
            color=h["color"],
            linewidth=h["linewidth"],
            linestyle=h["linestyle"],
            alpha=h["alpha"])

    # Ploting shaping lines
    if cumulative:
        pl.hlines(0.5, kwargs["xlim"][0], kwargs["xlim"][1], colors='0.6', linestyles=':', linewidth=2, alpha=0.5)
        pl.vlines(0, kwargs["ylim"][0], kwargs["ylim"][1], colors='0.6', linestyles=':', linewidth=2, alpha=0.5)

    # Tweak the graph
    kwargs["title"]="Density_Plot"
    _plot_postprocessing(kwargs)


#~~~~~~~ Generic plots ~~~~~~~#

def PCA_var_plot (df, variable_col, sample_col, value_col, plot_style="ggplot", **kwargs):
    """
    Plot a the frequence of contribution to the variance of the principal componants from a dataframe in “stacked” or “record” format
    *  df
        A pandas dataframe with a least the 3 following columns:
        - variable_col = column containing variable identifiers (such as gene identifiers)
        - sample_col = column containing sample identifiers (such as experimental condition)
        - value_col = column containing quantitative numeric values
    *  variable_col
        Name or index of the column containing variable identifiers
    *  sample_col
        Name or index of the column containing sample identifiers
    *  value_col
        Name or index of the column containing values
    *  plot_style
        Default plot style for pyplot ('grayscale'|'bmh'|'ggplot'|'dark_background'|'classic'|'fivethirtyeight'...)[ DEFAULT: "ggplot" ]
    *  kwargs
        Additional parameters for plot appearance derived from pylab basic plot arguments such as: title, xlabel, ylabel, color, alpha, fontsize, fontname
    """
    # pivot data to reorganize df
    df = df.pivot(index=variable_col, columns=sample_col, values=value_col)
        
    # Perform PCA analysis
    pca_res = pl.mlab.PCA(df, standardize=True)
    
    # transform and store the pca results in a dataframe
    pca_df = pd.DataFrame (columns=["comp","X", "Y"])
    for x, y in enumerate(pca_res.fracs):
        pca_df.at[len(pca_df)]= [ "PC{}".format(x+1), x+1, y*100]

    # Prepare plotting area  and parameters
    figsize = kwargs["figsize"] if "figsize" in kwargs else [len(pca_df)*2, 5]
    fontsize = kwargs["fontsize"] if "fontsize" in kwargs else 10
    fontname = kwargs["fontname"] if "fontname" in kwargs else "Sans"
    fig, ax = pl.subplots(figsize=figsize)
    pl.style.use(plot_style)
    
    # Plot the bar graph
    color =  kwargs["color"] if "color" in kwargs else None
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1
    ax.bar(pca_df.X,pca_df.Y, align='center', color=color, alpha=alpha)
    
    # Add the % for each component on the top of each bar
    for i, val in pca_df.iterrows():
        y_adj = val.Y+5 if val.Y<=90 else val.Y-5
        ax.text(x=val.X, y=y_adj, s="{}%".format(round(val.Y, 2)), horizontalalignment="center", fontsize = fontsize-2, fontname=fontname)
    
    # Tweak plot
    ax.set_ylabel(kwargs["ylabel"] if "ylabel" in kwargs else "% variance explained")
    ax.set_xlabel(kwargs["xlabel"] if "xlabel" in kwargs else "Principal components")
    ax.set_title(kwargs["title"] if "title" in kwargs else "Percentage of variance explained per principal component")
    ax.set_ylim((0,100))
    ax.set_xticks(pca_df.X)
    ax.set_xticklabels(pca_df.comp)
        
    # Fontsize and fontname 
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
        item.set_fontname(fontname)

    return (pca_df)


def PCA(df, variable_col, sample_col, value_col, pcx=1, pcy=2, point_label=False, plot_style="ggplot", **kwargs):
    """
    Plot a the frequence of contribution to the variance of the principal components from a dataframe in “stacked” or “record” format.
    Use matplotlib PCA implementation
    *  df
        A pandas dataframe with a least the 3 following columns:
        - variable_col = column containing variable identifiers (such as gene identifiers)
        - sample_col = column containing sample identifiers (such as experimental condition)
        - value_col = column containing quantitative numeric values
    *  variable_col
        Name or index of the column containing variable identifiers
    *  sample_col
        Name or index of the column containing sample identifiers
    *  value_col
        Name or index of the column containing values
    *  pcx
        Number of the component to plot on the x axis [ DEFAULT: 1 ]
    *  pcy
        Number of the component to plot on the y axis [ DEFAULT: 2 ]
    * point_label
        If True the points will be labelled with their names
    *  plot_style
        Default plot style for pyplot ('grayscale'|'bmh'|'ggplot'|'dark_background'|'classic'|'fivethirtyeight'...)[ DEFAULT: "ggplot" ]
    *  kwargs
        Additional parameters for plot appearance derived from pylab basic plot arguments such as: title, color, alpha, fontsize, fontname, linewidths
    """
    # pivot data to reorganize df
    df = df.pivot(index=variable_col, columns=sample_col, values=value_col)
    
    # Perform PCA analysis
    pca_res = pl.mlab.PCA(df, standardize=True)
    
    # Prepare plotting area and parameters
    figsize = kwargs["figsize"] if "figsize" in kwargs else [10, 10]
    fig, ax = pl.subplots(figsize=figsize)
    pl.style.use(plot_style)
    fontsize = kwargs["fontsize"] if "fontsize" in kwargs else 10
    fontname = kwargs["fontname"] if "fontname" in kwargs else "Sans"
    color =  kwargs["color"] if "color" in kwargs else "black"
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1
    linewidths = kwargs["linewidths"] if "linewidths" in kwargs else 3
    
    # Extract data and plot it
    df_res = pd.DataFrame(data=np.array(pca_res.Wt), index=pca_res.a.columns, columns=range(1, len(pca_res.a.columns)+1))
    ax.scatter(df_res[pcx], df_res[pcy], linewidths=linewidths, edgecolor=color, color=color, alpha=alpha)
    
    if point_label:
        for name, val in df_res.iterrows():
            ax.text(x=val[pcx], y=val[pcy], s=name, horizontalalignment="center", fontsize = fontsize-2, fontname=fontname)
    
    ax.set_xlabel("PC{} ({}%)".format(pcx, round(pca_res.fracs[pcx-1]*100, 2)))
    ax.set_ylabel("PC{} ({}%)".format(pcy, round(pca_res.fracs[pcy-1]*100, 2)))
    ax.set_title(kwargs["title"] if "title" in kwargs else "Principal component PC{}/PC{}".format(pcx, pcy))
    
    # Fontsize and fontname
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
        item.set_fontname(fontname)

    return (df_res)

def PCA2(df, variable_col, sample_col, value_col, pcx=1, pcy=2, point_label=False, plot_style="ggplot", **kwargs):
    """
    Plot a the frequence of contribution to the variance of the principal components from a dataframe in “stacked” or “record” format.
    Use sklearn PCA implementation
    *  df
        A pandas dataframe with a least the 3 following columns:
        - variable_col = column containing variable identifiers (such as gene identifiers)
        - sample_col = column containing sample identifiers (such as experimental condition)
        - value_col = column containing quantitative numeric values
    *  variable_col
        Name or index of the column containing variable identifiers
    *  sample_col
        Name or index of the column containing sample identifiers
    *  value_col
        Name or index of the column containing values
    *  pcx
        Number of the component to plot on the x axis [ DEFAULT: 1 ]
    *  pcy
        Number of the component to plot on the y axis [ DEFAULT: 2 ]
    * point_label
        If True the points will be labelled with their names
    *  plot_style
        Default plot style for pyplot ('grayscale'|'bmh'|'ggplot'|'dark_background'|'classic'|'fivethirtyeight'...)[ DEFAULT: "ggplot" ]
    *  kwargs
        Additional parameters for plot appearance derived from pylab basic plot arguments such as: title, color, alpha, fontsize, fontname, linewidths
    """
    
    from sklearn import decomposition
    
    # pivot data to reorganize df
    df = df.pivot(index=sample_col, columns=variable_col, values=value_col)
    n_comp = len(df)
    print (n_comp) 
    
    # Perform PCA analysis with sklearn
    pca = decomposition.PCA()
    pca_res = pca.fit(df)
    
    # Extract pca data and cast in panda dataframe
    pca_data = pd.DataFrame(pca_res.transform(df), index=df.index, columns=range(1,n_comp+1))
    
    # Prepare plotting area and parameters
    figsize = kwargs["figsize"] if "figsize" in kwargs else [10, 10]
    fig, ax = pl.subplots(figsize=figsize)
    pl.style.use(plot_style)
    fontsize = kwargs["fontsize"] if "fontsize" in kwargs else 10
    fontname = kwargs["fontname"] if "fontname" in kwargs else "Sans"
    color =  kwargs["color"] if "color" in kwargs else "black"
    alpha = kwargs["alpha"] if "alpha" in kwargs else 1
    linewidths = kwargs["linewidths"] if "linewidths" in kwargs else 3
    
    # Extract data and plot it
    for name, val in pca_data.iterrows():
        ax.scatter(val[pcx], val[pcy], linewidths=linewidths, edgecolor=color, color=color, alpha=alpha)
        if point_label:
            ax.text(x=val[pcx], y=val[pcy], s=name, horizontalalignment="center", fontsize=fontsize-2, fontname=fontname)

    ax.set_xlabel("PC{} ({}%)".format(pcx, round(pca_res.explained_variance_ratio_[pcx-1]*100, 2)))
    ax.set_ylabel("PC{} ({}%)".format(pcy, round(pca_res.explained_variance_ratio_[pcy-1]*100, 2)))
    ax.set_title(kwargs["title"] if "title" in kwargs else "Principal component PC{}/PC{})".format(pcx, pcy))
    
    # Fontsize and fontname
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
        item.set_fontname(fontname)

    return pca_data


#~~~~~~~ Generic utilities ~~~~~~~#

def get_color_list(n, gamma=1, colormap="brg"):
    """
    Return a list of l length with gradient colors from a given matplot lib colormap palette
    Before usage the palette can be tried with the *try_color_list* function
    *  n
        Number of color scalar in the list
    *  gamma
        Move the distribution toward the left (gamma<1) or the right (gamma>1)
    *  colormap
        colormap color palette from matplotlib package see http://matplotlib.org/examples/color/colormaps_reference.html
                        example : inferno magma hot blues cool spring winter brg ocean hsv jet ... [DEFAULT: brg]
    """
    
    # Init variables
    cmap = mpl.cm.get_cmap(colormap)
    cmap.set_gamma(gamma)
    
    index = 1 # skip the first value 
    n_col_cmap = cmap.N-2 # remove border colors
    step = int(n_col_cmap/(n-1)) if n > 1 else n_col_cmap/2
    
    # Create the list of colors
    cl = []
    for i in range (n):
        cl.append(cmap(index))
        index+=step

    # Yield colour and cycle from the start if needed
    i=0
    while True:
        yield(cl[i])
        i = i+1 if i < n-1 else 0


def try_color_list (n_color, n_values, gamma=1, colormap="brg",):
    """
    Test a palette generated by get_color_list
    * n_color
        Number of colour in the palette
    * n_values
        Number of values required (could be more that the number of colour but will cycle from the start)
    * gamma
        Move the distribution toward the left (gamma<1) or the right (gamma>1)
    * colormap
        Color palette from matplotlib package see http://matplotlib.org/examples/color/colormaps_reference.html
        Examples : inferno magma hot blues cool spring winter brg ocean hsv jet ... [DEFAULT: brg]
    """
    
    pl.figure(figsize=(n_values/2+1,1))
    pl.xlim(-1, n_values+1)
    pl.axis("off")
    
    colgen = get_color_list(n_color, gamma, colormap)
    for i in range(n_values):
        pl.scatter(i, 0, c=next(colgen), s=400, linewidth=0)
    

def plot_text (text, plot_len=20, align="center", **kwargs):
    """
    Plot a text alone as graph. Useful to separate series of data plots in interactive session.
    * text
        Test message to plot
    * len_plot
        Length of the plotting area [DEFAULT: 20]
    * align
        Alignment of the text ['left' | 'right' | 'center' ] [DEFAULT: 'center']
    * kwargs
        Additional parameters from matplotlib.text.Text class see http://matplotlib.org/users/text_intro.html
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
    if "fontsize" in kws:
        fontsize=kws["fontsize"]
    else:
        fontsize=10
    if "title" in kws:
        pl.title(kws["title"], fontsize=fontsize)
    if "xlabel" in kws:
        pl.xlabel(kws["xlabel"], fontsize=fontsize)
    if "ylabel" in kws:
        pl.ylabel(kws["ylabel"], fontsize=fontsize)
    if "xlim" in kws:
        pl.xlim(kws["xlim"])
    if "ylim" in kws:
        pl.ylim(kws["ylim"])
    pl.legend(
        bbox_to_anchor=(1, 1),
        loc=2,
        frameon=False,
        fontsize=fontsize)
    pl.xticks(fontsize=fontsize)
    pl.yticks(fontsize=fontsize)
    
def _parse_highlight_list(highlight_list, df, default_val={}, highlight_palette="Set1", FDR=None, FDR_col=None, min_targets=0):
    # Parse, clean and define default values if needed
    
    colors = get_color_list(n=len(highlight_list), colormap=highlight_palette)
    clean_list=[]
    
    for i, h in enumerate(highlight_list):
        
        # Extract of define default values = Can be expanded later
        if "df" not in h and "target_id" in h:
            df2 = df[(df.target_id.isin(h["target_id"]))]
        elif "df" in h:
            df2 = h["df"].dropna()
        else:
            print("Target_id list of dataframe required for series #{}. Skipping to the next one".format(i))
            continue
        
        if FDR:
            df2 = df2[(df2[FDR_col]<=FDR)]
        
        if len(df2) <= min_targets:
            #print("Not enought values in series #{}. Skipping to the next one".format(i))
            continue
        
        h2={
            "df":df2,
            "label":"{}  n={}".format(h.get("label", "Series #"+str(i)), len(df2)),
            "color":h.get("color", next(colors)),
            "alpha":h.get("alpha", default_val.get("alpha", 1)),
            "marker":h.get("marker", default_val.get("marker", "o")),
            "linewidth":h.get("linewidth", default_val.get("linewidth", 0)),
            "linestyle":h.get("linestyle", default_val.get("linestyle", "-"))}
        
        clean_list.append(h2)
    
    return clean_list



































