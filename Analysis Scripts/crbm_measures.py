import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


def compute_crbm(X_norm, Y_norm):
    """Compute CR and BM measures.

    Args:
        X_norm (array-like): Scaled hippocampal residuals.
        Y_norm (array-like): Scaled memory residuals.

    Returns:
        tuple: CR and BM values.
    """
    # Calculate CR values
    theta_flat = np.deg2rad(0.0)     
    theta_diag = np.deg2rad(45.0)       
    theta_perp = np.deg2rad(90.0)     
    kx, ky = 2, 2
    wX_norm = 0.5 * (1 + np.tanh(kx * X_norm))
    wY_norm = 0.5 * (1 + np.tanh(ky * Y_norm))
    theta_raw = ((1 - wX_norm) * (wY_norm) * theta_flat +
                wX_norm * (1 - wY_norm) * theta_perp +
                ((wX_norm * wY_norm) + (1 - wX_norm) * (1 - wY_norm)) * theta_diag)
    CR_raw = np.cos(theta_raw) * Y_norm - np.sin(theta_raw) * X_norm

    # Then normalize accordingly
    CR_bound = 1.0484785942244885 # pre-calculated max / min possible value for this dataset
    CR_raw = (CR_raw + CR_bound) / (2 * CR_bound)

    # Calculate BM values
    dist_bm_point_raw = np.sqrt((X_norm - 1)**2 + (Y_norm - 1)**2)
    dist_bm_line_raw  = np.abs(X_norm - Y_norm) / np.sqrt(2)
    BM_raw = 3.414214 - (dist_bm_point_raw + dist_bm_line_raw) # pre-calculated max BM value
    BM_raw = BM_raw / 3.414214 # Scale to 0-1

    return pd.DataFrame(CR_raw, columns=["CR_raw"]), pd.DataFrame(BM_raw, columns=["BM_raw"])


def mask_to_segments(x, y, mask):
    """Convert masked (x, y) data into contiguous line segments for plotting.
    Specifically, for data within range. 

    Args:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.
        mask (array-like of bool): Boolean array indicating which points are within range.

    Returns:
        list of np.ndarray: A list of 2D arrays, each representing a continuous line segment.
    """
    segs = []
    if mask.any():
        idx = np.where(np.diff(mask.astype(int)) != 0)[0] + 1
        parts = np.split(np.arange(len(x)), idx)
        for p in parts:
            if mask[p].all() and len(p) > 1:
                pts = np.column_stack([x[p], y[p]]).reshape(-1,1,2)
                segs.append(pts)
    return segs


def plot_cr_moderation(df_raw):
    """Plot how CR moderates the relationship between hippocampal and memory change.

    Args:
        df_raw (pd.DataFrame): DataFrame containing X, Y, and CR columns.
    
    Returns:
        fig: Matplotlib figure object.
    """
    # Define custom colors to match R
    custom_colors_cr = ["#f5d7b0", "#f5be27", "#fa8d00", "#bc507f", "#7E1891", "#1C0159"]
    cmap_cr = LinearSegmentedColormap.from_list("cr_cmap", custom_colors_cr, N=256)

    # Split CR into bins
    n_bins = 60
    df_raw['CR_bin'] = pd.qcut(df_raw['CR'], q=n_bins, labels=False)

    # Prepare lists for storing results
    records = []
    x_line = np.linspace(df_raw['X'].min(), df_raw['X'].max(), 100)

    # Make line segments for each bin with >=5 points
    for b in range(n_bins):
        grp = df_raw[df_raw['CR_bin'] == b]
        if len(grp) < 5:
            continue
        # Run linear regression
        a, c = np.polyfit(grp['X'], grp['Y'], 1)
        # Store the X range for each bin so lines are drawn only where data exist
        records.append((grp['CR'].mean(), a, c, grp['X'].min(), grp['X'].max()))

    # Create a df of fitted line parameters
    lines = pd.DataFrame(records, columns=['CR_mean','slope','intercept','x_min','x_max'])

    # Build line segments
    segments_in = []   # line portions within the data range
    segments_out = []  # predicted line portions
    colors_in = []
    colors_out = []

    for _, row in lines.iterrows():
        # Compute Y values across the full X range
        y_line = row['slope'] * x_line + row['intercept']

        # Identify where this bin actually has data using a mask
        mask_in = (x_line >= row['x_min']) & (x_line <= row['x_max'])
        mask_out = ~mask_in

        segs_in = mask_to_segments(x_line, y_line, mask_in)
        segs_out = mask_to_segments(x_line, y_line, mask_out)

        segments_in.extend(segs_in)
        segments_out.extend(segs_out)
        colors_in.extend([row['CR_mean']]*len(segs_in))
        colors_out.extend([row['CR_mean']]*len(segs_out))

    # Create the line collections
    norm = plt.Normalize(vmin=0, vmax=1)

    # Solid lines = actual data ranges
    lc_in = LineCollection([s[:,0,:] for s in segments_in],
                        cmap=cmap_cr, norm=norm,
                        linewidths=5, alpha=0.9)
    lc_in.set_array(np.array(colors_in))

    # Faint lines = extrapolated portions outside observed X range
    lc_out = LineCollection([s[:,0,:] for s in segments_out],
                            cmap=cmap_cr, norm=norm,
                            linewidths=3, alpha=0.25)
    lc_out.set_array(np.array(colors_out))

    # Plot the way CR moderates the x–y relationship
    fig, ax = plt.subplots(figsize=(8,6))
    ax.add_collection(lc_out)   
    ax.add_collection(lc_in)   
    ax.autoscale()
    ax.set_ylim(-4.0, 1.5)
    ax.set_xlabel("Hippocampal Annual Change (age-adjusted)")
    ax.set_ylabel("Memory Annual Change (age-adjusted)")
    cb = plt.colorbar(lc_in, ax=ax, label='Cognitive Reserve')
    cb.ax.tick_params(labelsize=9)
    ax.grid(alpha=0.3)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()

    return fig


def main():
    data = pd.read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Clean Data All Cohorts/Wide Format/Analysed data.csv")
    X_res = pd.to_numeric(data["res_hc_slopes_age"], errors="coerce")   # hippocampal residuals
    Y_res = pd.to_numeric(data["res_mem_slopes_age"],  errors="coerce")   # memory residuals

    # Drop NA
    m = X_res.notna() & Y_res.notna()
    X_res, Y_res = X_res[m], Y_res[m]

    # Z-score normalization
    X_zscore = (X_res - X_res.mean()) / X_res.std()
    Y_zscore = (Y_res - Y_res.mean()) / Y_res.std()

    # Normalize to [-1, 1]
    X_norm = X_zscore / max(abs(X_zscore.min(skipna=True)), abs(X_zscore.max(skipna=True)))
    Y_norm = Y_zscore / max(abs(Y_zscore.min(skipna=True)), abs(Y_zscore.max(skipna=True)))

    # Compute CR and BM
    CR_raw, BM_raw = compute_crbm(X_norm.values, Y_norm.values)

    # Add the new vars back into the df
    id_col = "id"                      
    ids = data.loc[m, id_col].reset_index(drop=True)

    df_raw = pd.DataFrame({
        id_col: ids,
        "X": X_norm.reset_index(drop=True),
        "Y": Y_norm.reset_index(drop=True),
        "CR": CR_raw["CR_raw"].reset_index(drop=True),
        "BM": BM_raw["BM_raw"].reset_index(drop=True),
    })

    # Save the data with new CRBM variables
    df_raw.to_csv("/tsd/p274/data/durable/projects/p027-cr_bm/crbm_data.csv", index=False)

    # Plot CR moderation
    fig = plot_cr_moderation(df_raw)
    fig_dir = "/tsd/p274/home/p274-rachelm/Documents/Fig4.png"
    fig.savefig(fig_dir, dpi=300, bbox_inches='tight', facecolor='white')


if __name__ == "__main__":
    main()