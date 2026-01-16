"""
Visualization utilities for earthquake analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from .config import (
        COLORS, SOIL_COLORS, HAZARD_COLORS, MOON_PHASE_COLORS,
        FIGURE_DPI, FIGURE_SIZE_DEFAULT, FIGURE_SIZE_LARGE, FIGURE_SIZE_WIDE,
        FONT_SIZE, TITLE_SIZE, LABEL_SIZE,
        CLUSTER_CMAP, DENSITY_CMAP, DEPTH_CMAP,
        TURKEY_BOUNDS, TURKEY_PROVINCES_GEOJSON
    )
except ImportError:
    from config import (
        COLORS, SOIL_COLORS, HAZARD_COLORS, MOON_PHASE_COLORS,
        FIGURE_DPI, FIGURE_SIZE_DEFAULT, FIGURE_SIZE_LARGE, FIGURE_SIZE_WIDE,
        FONT_SIZE, TITLE_SIZE, LABEL_SIZE,
        CLUSTER_CMAP, DENSITY_CMAP, DEPTH_CMAP,
        TURKEY_BOUNDS, TURKEY_PROVINCES_GEOJSON
    )

# Cache for Turkey GeoDataFrame
_turkey_gdf_cache = None


def setup_style():
    """Set up matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': FIGURE_SIZE_DEFAULT,
        'figure.dpi': FIGURE_DPI,
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'legend.fontsize': FONT_SIZE - 1,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
    })


def create_turkey_basemap(ax=None, title=None):
    """
    Create a basic Turkey map with boundaries.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    ax.set_xlim(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'])
    ax.set_ylim(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    return ax


def plot_earthquake_map(lons, lats, magnitudes=None, depths=None,
                         color_by='magnitude', ax=None, title=None,
                         alpha=0.6, size_scale=10):
    """
    Plot earthquake locations on Turkey map.

    Parameters
    ----------
    lons, lats : array-like
        Longitude and latitude coordinates
    magnitudes : array-like, optional
        Earthquake magnitudes for sizing/coloring
    depths : array-like, optional
        Earthquake depths for coloring
    color_by : str
        'magnitude', 'depth', or 'category'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    alpha : float
        Point transparency
    size_scale : float
        Base size multiplier for points

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    ax = create_turkey_basemap(ax, title)

    # Determine sizes
    if magnitudes is not None:
        sizes = (np.asarray(magnitudes) ** 2) * size_scale
    else:
        sizes = 20

    # Determine colors
    if color_by == 'magnitude' and magnitudes is not None:
        scatter = ax.scatter(lons, lats, c=magnitudes, s=sizes,
                            cmap='YlOrRd', alpha=alpha, edgecolors='white',
                            linewidth=0.3)
        plt.colorbar(scatter, ax=ax, label='Magnitude', shrink=0.8)
    elif color_by == 'depth' and depths is not None:
        scatter = ax.scatter(lons, lats, c=depths, s=sizes,
                            cmap=DEPTH_CMAP, alpha=alpha, edgecolors='white',
                            linewidth=0.3)
        plt.colorbar(scatter, ax=ax, label='Depth (km)', shrink=0.8)
    else:
        ax.scatter(lons, lats, s=sizes, c=COLORS['earthquake'],
                  alpha=alpha, edgecolors='white', linewidth=0.3)

    return ax


def plot_kde_density(lons, lats, ax=None, title=None, levels=15,
                     cmap=DENSITY_CMAP, fill=True):
    """
    Plot KDE density contours for earthquake distribution.

    Parameters
    ----------
    lons, lats : array-like
        Longitude and latitude coordinates
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    levels : int
        Number of contour levels
    cmap : str
        Colormap name
    fill : bool
        Whether to fill contours

    Returns
    -------
    tuple
        (ax, kde) - axes object and KDE object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    ax = create_turkey_basemap(ax, title)

    # Create KDE
    positions = np.vstack([lons, lats])
    kde = stats.gaussian_kde(positions, bw_method='scott')

    # Create grid
    xi = np.linspace(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'], 100)
    yi = np.linspace(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'], 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Evaluate KDE
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    # Plot
    if fill:
        contour = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, alpha=0.8)
        plt.colorbar(contour, ax=ax, label='Density', shrink=0.8)
    else:
        contour = ax.contour(Xi, Yi, Zi, levels=levels, cmap=cmap)
        ax.clabel(contour, inline=True, fontsize=8)

    return ax, kde


def plot_fault_lines(fault_df, ax=None, color='black', linewidth=1.5, label=True):
    """
    Plot fault lines on map.

    Parameters
    ----------
    fault_df : pandas.DataFrame
        Fault data with lat_start, lon_start, lat_end, lon_end columns
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    color : str
        Line color
    linewidth : float
        Line width
    label : bool
        Whether to add fault name labels

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    for idx, row in fault_df.iterrows():
        ax.plot([row['lon_start'], row['lon_end']], [row['lat_start'], row['lat_end']],
                color=color, linewidth=linewidth, zorder=5)

        if label and 'fault_name' in row:
            mid_lon = (row['lon_start'] + row['lon_end']) / 2
            mid_lat = (row['lat_start'] + row['lat_end']) / 2
            ax.annotate(row['fault_name'], (mid_lon, mid_lat),
                       fontsize=7, ha='center', alpha=0.7)

    return ax


def plot_gutenberg_richter(magnitudes, mc, b_result, ax=None, title=None):
    """
    Plot Gutenberg-Richter frequency-magnitude distribution.

    Parameters
    ----------
    magnitudes : array-like
        Earthquake magnitudes
    mc : float
        Magnitude of completeness
    b_result : dict
        Result from gutenberg_richter_bvalue function
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    magnitudes = np.asarray(magnitudes)

    # Create magnitude bins
    mag_bins = np.arange(np.floor(magnitudes.min()), np.ceil(magnitudes.max()) + 0.1, 0.1)
    bin_centers = (mag_bins[:-1] + mag_bins[1:]) / 2

    # Calculate cumulative counts
    cum_counts = np.array([np.sum(magnitudes >= m) for m in bin_centers])

    # Plot observed data
    ax.scatter(bin_centers, cum_counts, c=COLORS['primary'], s=30, alpha=0.6,
               label='Observed', zorder=3)

    # Plot Mc line
    ax.axvline(mc, color=COLORS['warning'], linestyle='--', linewidth=2,
               label=f'Mc = {mc:.1f}', zorder=2)

    # Plot fitted G-R line
    if b_result is not None:
        mag_fit = np.linspace(mc, magnitudes.max(), 50)
        log_n_fit = b_result['a'] - b_result['b'] * mag_fit
        n_fit = 10 ** log_n_fit

        ax.plot(mag_fit, n_fit, color=COLORS['danger'], linewidth=2,
                label=f'G-R fit (b={b_result["b"]:.2f})', zorder=4)

    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Cumulative Number (N >= M)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    else:
        ax.set_title('Gutenberg-Richter Frequency-Magnitude Distribution',
                    fontsize=TITLE_SIZE, fontweight='bold')

    return ax


def plot_seismic_gap(segment_data, fault_name, ax=None, title=None):
    """
    Plot seismic activity along a fault line.

    Parameters
    ----------
    segment_data : pandas.DataFrame
        Segment data with distance_km and earthquake_count columns
    fault_name : str
        Name of the fault
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    # Bar plot of earthquake counts
    bars = ax.bar(segment_data['distance_km'], segment_data['earthquake_count'],
                  width=segment_data['segment_length'].iloc[0] * 0.9,
                  color=COLORS['primary'], alpha=0.7, edgecolor='white')

    # Highlight gaps (low activity segments)
    if 'is_gap' in segment_data.columns:
        gap_segments = segment_data[segment_data['is_gap']]
        for idx, row in gap_segments.iterrows():
            ax.axvspan(row['distance_km'] - row['segment_length']/2,
                      row['distance_km'] + row['segment_length']/2,
                      color=COLORS['danger'], alpha=0.3)

    # Mean line
    mean_count = segment_data['earthquake_count'].mean()
    ax.axhline(mean_count, color=COLORS['secondary'], linestyle='--',
               linewidth=2, label=f'Mean = {mean_count:.1f}')

    ax.set_xlabel('Distance Along Fault (km)')
    ax.set_ylabel('Earthquake Count')
    ax.legend()

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    else:
        ax.set_title(f'Seismic Activity Along {fault_name}',
                    fontsize=TITLE_SIZE, fontweight='bold')

    return ax


def plot_time_series(times, values, ax=None, title=None, xlabel='Date',
                     ylabel='Value', color=None, rolling_window=None):
    """
    Plot time series with optional rolling average.

    Parameters
    ----------
    times : array-like
        Time values (datetime)
    values : array-like
        Data values
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    xlabel, ylabel : str
        Axis labels
    color : str, optional
        Line color
    rolling_window : int, optional
        Window size for rolling average

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    if color is None:
        color = COLORS['primary']

    ax.plot(times, values, color=color, alpha=0.6, linewidth=0.8)

    if rolling_window:
        import pandas as pd
        rolling = pd.Series(values).rolling(window=rolling_window).mean()
        ax.plot(times, rolling, color=COLORS['danger'], linewidth=2,
                label=f'{rolling_window}-point Moving Average')
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()

    return ax


def plot_cumulative(times, values, ax=None, title=None, ylabel='Cumulative Value',
                    color=None, log_scale=False):
    """
    Plot cumulative values over time.

    Parameters
    ----------
    times : array-like
        Time values (datetime)
    values : array-like
        Data values to cumulate
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    ylabel : str
        Y-axis label
    color : str, optional
        Line color
    log_scale : bool
        Whether to use log scale on y-axis

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    if color is None:
        color = COLORS['primary']

    cumulative = np.cumsum(values)
    ax.plot(times, cumulative, color=color, linewidth=2)
    ax.fill_between(times, 0, cumulative, alpha=0.3, color=color)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()

    return ax


def plot_histogram_with_fit(data, distribution='normal', bins=50, ax=None,
                            title=None, xlabel='Value', ylabel='Frequency'):
    """
    Plot histogram with fitted distribution.

    Parameters
    ----------
    data : array-like
        Data to plot
    distribution : str
        Distribution to fit: 'normal', 'exponential', 'gamma', 'weibull'
    bins : int
        Number of histogram bins
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    xlabel, ylabel : str
        Axis labels

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    data = np.asarray(data)

    # Plot histogram
    counts, bin_edges, _ = ax.hist(data, bins=bins, density=True, alpha=0.7,
                                    color=COLORS['primary'], edgecolor='white')

    # Fit distribution
    x = np.linspace(data.min(), data.max(), 200)

    if distribution == 'normal':
        mu, sigma = stats.norm.fit(data)
        y = stats.norm.pdf(x, mu, sigma)
        label = f'Normal ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})'
    elif distribution == 'exponential':
        loc, scale = stats.expon.fit(data)
        y = stats.expon.pdf(x, loc, scale)
        label = f'Exponential ($\\lambda$={1/scale:.4f})'
    elif distribution == 'gamma':
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        y = stats.gamma.pdf(x, shape, loc, scale)
        label = f'Gamma (k={shape:.2f}, $\\theta$={scale:.2f})'
    elif distribution == 'weibull':
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)
        y = stats.weibull_min.pdf(x, shape, loc, scale)
        label = f'Weibull (k={shape:.2f}, $\\lambda$={scale:.2f})'
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    ax.plot(x, y, color=COLORS['danger'], linewidth=2, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    return ax


def plot_correlation_matrix(corr_matrix, labels=None, ax=None, title=None,
                            cmap='RdBu_r', annotate=True):
    """
    Plot correlation matrix as heatmap.

    Parameters
    ----------
    corr_matrix : array-like
        Correlation matrix
    labels : list, optional
        Variable labels
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    cmap : str
        Colormap name
    annotate : bool
        Whether to show correlation values

    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    if annotate:
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=8,
                       color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    return ax


def save_figure(fig, filename, folder=None, dpi=FIGURE_DPI, formats=['png']):
    """
    Save figure to file(s).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    folder : str, optional
        Output folder path
    dpi : int
        Output resolution
    formats : list
        List of output formats (e.g., ['png', 'pdf'])
    """
    import os
    try:
        from .config import FIGURES_PATH
    except ImportError:
        from config import FIGURES_PATH

    if folder is None:
        folder = FIGURES_PATH

    os.makedirs(folder, exist_ok=True)

    for fmt in formats:
        filepath = os.path.join(folder, f"{filename}.{fmt}")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")


# =============================================================================
# TURKEY MAP FUNCTIONS WITH GEOPANDAS
# =============================================================================

def load_turkey_provinces(geojson_path=None):
    """
    Load Turkey provinces GeoDataFrame with caching.

    Parameters
    ----------
    geojson_path : str, optional
        Path to Turkey provinces GeoJSON file.
        If None, uses TURKEY_PROVINCES_GEOJSON from config.

    Returns
    -------
    geopandas.GeoDataFrame
        Turkey provinces with geometry
    """
    global _turkey_gdf_cache

    if _turkey_gdf_cache is not None:
        return _turkey_gdf_cache

    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for Turkey map functions. Install with: pip install geopandas")

    if geojson_path is None:
        geojson_path = TURKEY_PROVINCES_GEOJSON

    _turkey_gdf_cache = gpd.read_file(geojson_path)
    return _turkey_gdf_cache


def normalize_province_name(name):
    """
    Normalize Turkish province names for matching.

    Handles encoding issues and common variations between Turkish and ASCII.
    """
    if name is None:
        return None

    # Convert to uppercase for comparison
    name = str(name).strip().upper()

    # Common replacements for Turkish characters (both directions)
    replacements = {
        'İ': 'I', 'Ş': 'S', 'Ğ': 'G', 'Ü': 'U', 'Ö': 'O', 'Ç': 'C',
        'ı': 'I', 'ş': 'S', 'ğ': 'G', 'ü': 'U', 'ö': 'O', 'ç': 'C',
        'Ä±': 'I',  # Unicode variant
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    # Special mappings for known variations (after character replacement)
    name_mappings = {
        # GeoJSON name: standard name (uppercase after char replacement)
        'AFYONKARAHISAR': 'AFYON',
        'AFYON': 'AFYON',
        'KAHRAMANMARAS': 'K.MARAS',
        'K.MARAS': 'K.MARAS',
        'KINKKALE': 'KIRIKKALE',
        'KIRIKKALE': 'KIRIKKALE',
        'SANLIURFA': 'SANLIURFA',
        'ZINGULDAK': 'ZONGULDAK',
        'ZONGULDAK': 'ZONGULDAK',
        'SIRNAK': 'SIRNAK',
        'GUMUSHANE': 'GUMUSHANE',
        'NEVSEHIR': 'NEVSEHIR',
        'KIRSEHIR': 'KIRSEHIR',
        'CANKIRI': 'CANKIRI',
    }

    return name_mappings.get(name, name)


def create_turkey_basemap_geo(ax=None, figsize=None, title=None, show_labels=False):
    """
    Create a Turkey basemap with province boundaries using geopandas.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, optional
        Figure size. Defaults to FIGURE_SIZE_DEFAULT.
    title : str, optional
        Plot title
    show_labels : bool
        Whether to show province name labels

    Returns
    -------
    tuple
        (ax, gdf) - axes and GeoDataFrame
    """
    gdf = load_turkey_provinces()

    if ax is None:
        if figsize is None:
            figsize = FIGURE_SIZE_DEFAULT
        fig, ax = plt.subplots(figsize=figsize)

    # Plot province boundaries
    gdf.boundary.plot(ax=ax, linewidth=0.5, color='gray', alpha=0.7)

    # Add province labels if requested
    if show_labels:
        for idx, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(row['NAME_1'], xy=(centroid.x, centroid.y),
                       ha='center', va='center', fontsize=6, alpha=0.6)

    ax.set_xlim(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'])
    ax.set_ylim(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    return ax, gdf


def create_turkey_choropleth(data_df, value_column, province_column='province',
                              cmap='YlOrRd', ax=None, figsize=None, title=None,
                              legend_label=None, edgecolor='white', linewidth=0.5,
                              missing_color='lightgray', vmin=None, vmax=None,
                              show_labels=False):
    """
    Create a choropleth map of Turkey provinces.

    Parameters
    ----------
    data_df : pandas.DataFrame
        DataFrame containing province data
    value_column : str
        Column name with values to visualize
    province_column : str
        Column name with province names (default: 'province')
    cmap : str
        Matplotlib colormap name
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    legend_label : str, optional
        Colorbar label
    edgecolor : str
        Province boundary color
    linewidth : float
        Province boundary line width
    missing_color : str
        Color for provinces with no data
    vmin, vmax : float, optional
        Value range for colormap
    show_labels : bool
        Whether to show province names

    Returns
    -------
    tuple
        (fig, ax) - figure and axes
    """
    gdf = load_turkey_provinces()

    if figsize is None:
        figsize = FIGURE_SIZE_DEFAULT

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Normalize province names for matching
    data_df = data_df.copy()
    data_df['_norm_province'] = data_df[province_column].apply(normalize_province_name)

    gdf = gdf.copy()
    gdf['_norm_province'] = gdf['NAME_1'].apply(normalize_province_name)

    # Merge data with GeoDataFrame
    merged = gdf.merge(data_df[['_norm_province', value_column]],
                       on='_norm_province', how='left')

    # Plot base layer (missing data) - only if there are missing values
    missing_data = merged[merged[value_column].isna()]
    if len(missing_data) > 0:
        missing_data.plot(
            ax=ax, color=missing_color, edgecolor=edgecolor, linewidth=linewidth
        )

    # Plot choropleth
    merged_with_data = merged[merged[value_column].notna()]
    if len(merged_with_data) > 0:
        merged_with_data.plot(
            column=value_column, cmap=cmap, ax=ax,
            edgecolor=edgecolor, linewidth=linewidth,
            legend=True, vmin=vmin, vmax=vmax,
            legend_kwds={'label': legend_label or value_column, 'shrink': 0.8}
        )

    # Add province labels if requested
    if show_labels:
        for idx, row in merged.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(row['NAME_1'], xy=(centroid.x, centroid.y),
                       ha='center', va='center', fontsize=5, alpha=0.7)

    ax.set_xlim(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'])
    ax.set_ylim(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    return fig, ax


def plot_earthquakes_on_turkey_map(eq_df, lon_col='longitude', lat_col='latitude',
                                    mag_col='magnitude', depth_col='depth',
                                    color_by='magnitude', ax=None, figsize=None,
                                    title=None, alpha=0.6, size_scale=10,
                                    show_boundaries=True):
    """
    Plot earthquake scatter on Turkey map with province boundaries.

    Parameters
    ----------
    eq_df : pandas.DataFrame
        Earthquake data
    lon_col, lat_col : str
        Column names for coordinates
    mag_col : str
        Column name for magnitude (for sizing)
    depth_col : str
        Column name for depth (for coloring)
    color_by : str
        'magnitude', 'depth', or None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    alpha : float
        Point transparency
    size_scale : float
        Size multiplier for points
    show_boundaries : bool
        Whether to show province boundaries

    Returns
    -------
    tuple
        (fig, ax)
    """
    if figsize is None:
        figsize = FIGURE_SIZE_DEFAULT

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot Turkey base map with boundaries
    if show_boundaries:
        gdf = load_turkey_provinces()
        gdf.plot(ax=ax, color='#f5f5f5', edgecolor='gray', linewidth=0.3, alpha=0.8)

    # Get data
    lons = eq_df[lon_col].values
    lats = eq_df[lat_col].values

    # Determine sizes
    if mag_col and mag_col in eq_df.columns:
        sizes = (eq_df[mag_col].values ** 2) * size_scale
    else:
        sizes = 20

    # Determine colors and plot
    if color_by == 'magnitude' and mag_col in eq_df.columns:
        scatter = ax.scatter(lons, lats, c=eq_df[mag_col], s=sizes,
                            cmap='YlOrRd', alpha=alpha, edgecolors='white',
                            linewidth=0.3, zorder=5)
        plt.colorbar(scatter, ax=ax, label='Magnitude', shrink=0.8)
    elif color_by == 'depth' and depth_col in eq_df.columns:
        scatter = ax.scatter(lons, lats, c=eq_df[depth_col], s=sizes,
                            cmap=DEPTH_CMAP, alpha=alpha, edgecolors='white',
                            linewidth=0.3, zorder=5)
        plt.colorbar(scatter, ax=ax, label='Depth (km)', shrink=0.8)
    else:
        ax.scatter(lons, lats, s=sizes, c=COLORS['earthquake'],
                  alpha=alpha, edgecolors='white', linewidth=0.3, zorder=5)

    ax.set_xlim(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'])
    ax.set_ylim(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    return fig, ax


def plot_province_categories(data_df, category_column, province_column='province',
                              color_map=None, ax=None, figsize=None, title=None,
                              legend_title=None, edgecolor='white', linewidth=0.5,
                              show_labels=False):
    """
    Plot Turkey map with categorical province data.

    Parameters
    ----------
    data_df : pandas.DataFrame
        DataFrame containing province data
    category_column : str
        Column name with categorical values
    province_column : str
        Column name with province names
    color_map : dict, optional
        Mapping from category to color
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    legend_title : str, optional
        Legend title
    edgecolor : str
        Province boundary color
    linewidth : float
        Province boundary line width
    show_labels : bool
        Whether to show province names

    Returns
    -------
    tuple
        (fig, ax)
    """
    gdf = load_turkey_provinces()

    if figsize is None:
        figsize = FIGURE_SIZE_DEFAULT

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Normalize province names
    data_df = data_df.copy()
    data_df['_norm_province'] = data_df[province_column].apply(normalize_province_name)

    gdf = gdf.copy()
    gdf['_norm_province'] = gdf['NAME_1'].apply(normalize_province_name)

    # Merge
    merged = gdf.merge(data_df[['_norm_province', category_column]],
                       on='_norm_province', how='left')

    # Get unique categories
    categories = merged[category_column].dropna().unique()

    # Generate color map if not provided
    if color_map is None:
        cmap = plt.cm.get_cmap('tab10')
        color_map = {cat: cmap(i / len(categories)) for i, cat in enumerate(categories)}

    # Plot each category
    for cat in categories:
        subset = merged[merged[category_column] == cat]
        subset.plot(ax=ax, color=color_map.get(cat, 'gray'),
                   edgecolor=edgecolor, linewidth=linewidth)

    # Plot missing data
    missing = merged[merged[category_column].isna()]
    if len(missing) > 0:
        missing.plot(ax=ax, color='lightgray', edgecolor=edgecolor,
                    linewidth=linewidth)

    # Create custom legend with patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map.get(cat, 'gray'), edgecolor=edgecolor,
                             label=cat) for cat in categories]
    if len(missing) > 0:
        legend_elements.append(Patch(facecolor='lightgray', edgecolor=edgecolor, label='No Data'))

    ax.legend(handles=legend_elements, title=legend_title or category_column,
              loc='lower right', fontsize=8, title_fontsize=9)

    # Add province labels if requested
    if show_labels:
        for idx, row in merged.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(row['NAME_1'], xy=(centroid.x, centroid.y),
                       ha='center', va='center', fontsize=5, alpha=0.7)

    ax.set_xlim(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'])
    ax.set_ylim(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    return fig, ax


def plot_kde_on_turkey_map(lons, lats, ax=None, figsize=None, title=None,
                            levels=15, cmap=DENSITY_CMAP, fill=True,
                            show_boundaries=True, alpha=0.7):
    """
    Plot KDE density contours on Turkey map with province boundaries.

    Parameters
    ----------
    lons, lats : array-like
        Longitude and latitude coordinates
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    levels : int
        Number of contour levels
    cmap : str
        Colormap name
    fill : bool
        Whether to fill contours
    show_boundaries : bool
        Whether to show province boundaries
    alpha : float
        Contour transparency

    Returns
    -------
    tuple
        (fig, ax, kde)
    """
    if figsize is None:
        figsize = FIGURE_SIZE_DEFAULT

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot Turkey boundaries first (behind KDE)
    if show_boundaries:
        gdf = load_turkey_provinces()
        gdf.boundary.plot(ax=ax, linewidth=0.5, color='gray', alpha=0.5, zorder=1)

    # Create KDE
    positions = np.vstack([lons, lats])
    kde = stats.gaussian_kde(positions, bw_method='scott')

    # Create grid
    xi = np.linspace(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'], 100)
    yi = np.linspace(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'], 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Evaluate KDE
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    # Plot
    if fill:
        contour = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, alpha=alpha, zorder=2)
        plt.colorbar(contour, ax=ax, label='Density', shrink=0.8)
    else:
        contour = ax.contour(Xi, Yi, Zi, levels=levels, cmap=cmap, zorder=2)
        ax.clabel(contour, inline=True, fontsize=8)

    ax.set_xlim(TURKEY_BOUNDS['lon_min'], TURKEY_BOUNDS['lon_max'])
    ax.set_ylim(TURKEY_BOUNDS['lat_min'], TURKEY_BOUNDS['lat_max'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')

    return fig, ax, kde
