import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _maxabs(a):
    """Rescale an array to the fixed range [-1, 1].

    Args:
        a: input values.

    Returns:
        ndarray: Values linearly mapped from −1 to +1.
    """
    a = np.asarray(a, float)
    return a / (max(abs(a.min()), abs(a.max())))


def _zscore(a):
    """Z-scores array.

    Args:
        a: input values.

    Returns:
        ndarray: Z-scored values.
    """
    return (a - a.mean()) / (a.std())


def compute_CR(X, Y, cr_min, cr_max, scale='maxabs', k=2, theta_flat=np.deg2rad(0), 
               theta_diag=np.deg2rad(45), theta_perp=np.deg2rad(90)):
    """Compute the CR value for given X and Y coordinates.

    Test different k values and thetas. 

    Args:
        X, Y: Hippocampal and memory change values.
        cr_min: Pre-calculated min for normalization. 
        cr_max: Pre-calculated max for normalization.
            These are needed because different methods have different min-max.
        scale: {'maxabs', 'zscore'}, Pre-scaling strategy. 'maxabs' rescales to [−1,1]; 'zscore' 
            standardizes then rescales to [−1,1].
        theta_flat, theta_diag, theta_perp (float, optional): Orientation angles (radians) for each quadrant configuration.
        k (float, optional): Smoothing term for the tanh weighting functions.

    Returns:
        ndarray: CR values normalized to [0,1].
    """
    if scale == 'maxabs':
        Xn, Yn = _maxabs(X), _maxabs(Y)
    elif scale == 'zscore_maxabs':
        Xn, Yn = _zscore(X), _zscore(Y)
        Xn, Yn = _maxabs(Xn), _maxabs(Yn)
    else:
        Xn, Yn = X.copy(), Y.copy()

    k = k
    wx = 0.5 * (1 + np.tanh(k * Xn))
    wy = 0.5 * (1 + np.tanh(k * Yn))
    theta = (1 - wx) * wy * theta_flat + wx * (1 - wy) * theta_perp + ((wx * wy) + (1 - wx) * (1 - wy)) * theta_diag
    CR = np.cos(theta) * Yn - np.sin(theta) * Xn

    # Normalize and return
    return (CR - cr_min) / (cr_max - cr_min)


def compute_BM(X, Y, bm_max, scale='maxabs', alpha_point=0.5, metric='euclid'):
    """Compute the BM value for given X and Y coordinates.

    Tests euclidean distance (direct distance), manhattan distance (grid distance), 
    and chebyshev distance (max axis distance) on -1 to 1 scaled data or z-scored data.
    Also tests different weightings between point distance and line distance (alpha_point).
    When alpha_point=1, only point distance is used; when alpha_point=0, only line distance is used.

    Args:
        X, Y: Hippocampal and memory change values.
        bm_max: Pre-calculated max for normalization. Note the min is always 0.
            This is needed because different methods have different maxes.
        scale: {'maxabs', 'zscore'}, Pre-scaling strategy. 'maxabs' rescales to [−1,1]; 'zscore' 
            standardizes then rescales to [−1,1].
        alpha_point (float): Weighting factor between point distance and line distance (0 to 1).
        metric: {'euclid', 'manhattan', 'cheby'}, Distance metric to use.   
    
    Returns:
        ndarray: BM values normalized to [0,1].
    """
    if scale == 'maxabs':
        Xn, Yn = _maxabs(X), _maxabs(Y)
    elif scale == 'zscore_maxabs':
        Xn, Yn = _zscore(X), _zscore(Y)
        Xn, Yn = _maxabs(Xn), _maxabs(Yn)
    else:
        Xn, Yn = X.copy(), Y.copy()

    if metric == 'euclid':
        d_point = np.sqrt((Xn - 1)**2 + (Yn - 1)**2)
        d_line = np.abs(Xn - Yn) / np.sqrt(2)
    elif metric == 'manhattan':
        d_point = np.abs(Xn - 1) + np.abs(Yn - 1)
        d_line = np.abs(Xn - Yn)
    elif metric == 'cheby':
        d_point = np.maximum(np.abs(Xn - 1), np.abs(Yn - 1))
        d_line = np.abs(Xn - Yn) / 2
    else:
        raise ValueError("metric must be 'euclid', 'manhattan', or 'cheby'")

    # Combine both distances with consideration of alpha weighting
    BM = 2 * (alpha_point * d_point) + 2 * (1 - alpha_point) * d_line

    # Inverse scores and normalize
    BM = bm_max - BM
    return BM / bm_max


def compute_BM_data_based(X, Y):
    """Compute BM based on data-driven reference:
      1. Perpendicular distance from (X, Y) to the line y = 0.1437x
      2. Shortest distance from (X, Y) to the intersection of that line with x = 1
         (which is (1, 0.1437))
    
    Args:
        X, Y: Hippocampal and memory change values.

    Returns:
        ndarray: BM values normalized to [0,1].

    Note: ax+by+c=0 is the general form of a linear equation.
    So in this plane we can write mx - y + 0 = 0 because y = mx + 0.
    In this case, A = m, B = -1, C = 0 (extracted from above).
    """
    # First z-score and max-abs to [-1, 1]
    Xn, Yn = _zscore(X), _zscore(Y)
    Xn, Yn = _maxabs(Xn), _maxabs(Yn)

    # Distance to line y = m*x 
    m = 0.1437
    A, B, C = m, -1, 0
    dist_line = np.abs(A * Xn + B * Yn + C) / np.sqrt(A**2 + B**2) # Standard euclidean distance formula

    # Distance to the intersection point (1, 0.1437)
    x_ref = 1
    y_ref = m * x_ref
    dist_point = np.sqrt((Xn - x_ref)**2 + (Yn - y_ref)**2)

    # Combine both distances (same as theoretical BM logic)
    BM = dist_point + dist_line

    # Reverse so that higher BM means better maintenance
    BM = 3.3044 - BM # pre-defined max for this dataset
    return BM / 3.3044


def plot_panel(ax, X, Y, Z, title, label_text, cmap=None):
    """Plot a single panel of the CR or BM variation.
    
    Args:
        ax: Matplotlib Axes object to plot on.
        X, Y: Hippocampal and memory change values.
        Z: Computed CR or BM values on the grid.
        title: Title for the panel.
        label_text: Text label for the colorbar.
        cmap: Colormap to use for the surface.
    """
    mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
    levels = np.linspace(0, 1, 15)
    contours = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.6, alpha=1)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    ax.axhline(0, color='white', linewidth=1.0, alpha=0.8)
    ax.axvline(0, color='white', linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Hippocampal Annual Change (age-adjusted)")
    ax.set_ylabel("Memory Annual Change (age-adjusted)")
    ax.set_title(title, fontsize=10)
    return mesh, label_text


def main():
    # Read in data
    crbm_data = pd.read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/crbm_data.csv")
    mem_hc_data = pd.read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Clean Data All Cohorts/Wide Format/Analysed data.csv")
    id_col = "id"

    # Merge to align rows by id
    df = mem_hc_data[[id_col, "res_hc_slopes_age", "res_mem_slopes_age"]].merge(
            crbm_data[[id_col, "CR", "BM"]],
            on=id_col,
            how="inner"
        )

    # Build arrays
    X_res = pd.to_numeric(mem_hc_data["res_hc_slopes_age"], errors="coerce")   # hippocampal residuals
    Y_res = pd.to_numeric(mem_hc_data["res_mem_slopes_age"],  errors="coerce")   # memory residuals
    CR_original = pd.to_numeric(df["CR"], errors="coerce")
    BM_original = pd.to_numeric(df["BM"], errors="coerce")

    # Drop NA
    m = X_res.notna() & Y_res.notna()
    X_res, Y_res = X_res[m], Y_res[m]

    variants = []

    # A) Scaling method
    variants += [('CR scale=maxabs', dict(scale='maxabs', k=2), 'CR')]
    variants += [('BM scale=maxabs', dict(scale='maxabs', alpha_point=0.5, metric='euclid'), 'BM')]

    # B) Smoothness 
    for k in [1, 4, 6]:
        variants += [(f'CR k={k}', dict(scale='zscore_maxabs', k=k), 'CR')]

    # C) BM weighting
    for a in [0.3, 0.7]:
        variants += [(f'BM alpha_point={a}', dict(scale='zscore_maxabs', alpha_point=a, metric='euclid'), 'BM')]

    # D) BM distance metric
    for m in ['manhattan', 'cheby']:
        variants += [(f'BM metric={m}', dict(scale='zscore_maxabs', alpha_point=0.5, metric=m), 'BM')]

    # E) CR different thetas
    variants += [('CR thetas=0,22.5,45', dict(scale='zscore_maxabs', k=2,
                                            theta_flat=np.deg2rad(0),
                                            theta_diag=np.deg2rad(22.5),
                                            theta_perp=np.deg2rad(45)), 'CR')]
    variants += [('CR thetas=-15,45,105', dict(scale='zscore_maxabs', k=2,
                                            theta_flat=np.deg2rad(-15),
                                            theta_diag=np.deg2rad(45),
                                            theta_perp=np.deg2rad(105)), 'CR')]

    # F) BM data-based distance
    variants += [('BM data-based', dict(), 'BM')]

    # Define pre-calculated CR bounds 
    # This is so the actual data are min-maxed using the max possible for the variant 
    # rather than the observed max in the sample
    cr_bounds = [
        (-1.0484761307493822,  1.0484761307493822),  # CR scale=maxabs
        (-1.168672389055221,   1.168672389055221),   # CR k=1
        (-1.0190570698237502,  1.0190570698237502),  # CR k=4
        (-1.0113112142837763,  1.0113112142837763),  # CR k=6
        (-1.4140724591730809,  1.0265229491609051),  # CR thetas=0,22.5,45 (not symmetric)
        (-1.0130764092453262,  1.0130764092453262)   # CR thetas=-15,45,105
    ]

    bm_maxes = [
        3.414213562370681,    # BM α=0.5 Euclidean
        3.1798989873208354,   # BM α=0.3 Euclidean
        3.9597979746435166,   # BM α=0.7 Euclidean
        4.0,                  # BM Manhattan
        3.0,                  # BM Chebyshev
        3.3044                # BM data-based
    ]
    
    rows = []
    cr_idx = 0
    bm_idx = 0

    for name, params, kind in variants:
        if kind == 'CR':
            min, max = cr_bounds[cr_idx]
            arr = compute_CR(X_res, Y_res, cr_max=max, cr_min=min, **params)
            cr_idx += 1
            rho, p = spearmanr(CR_original, arr)
            rows.append([name, 'CR', rho, p])
        else:
            if name == 'BM data-based':
                arr = compute_BM_data_based(X_res, Y_res)
            else:
                max = bm_maxes[bm_idx]
                arr = compute_BM(X_res, Y_res, bm_max=max, **params)
                bm_idx += 1

            rho, p = spearmanr(BM_original, arr)
            rows.append([name, 'BM', rho, p])

    sens = pd.DataFrame(rows, columns=['variant', 'metric', 'spearman_vs_original', 'p_value'])
    sens['p_value'] = sens['p_value'].apply(lambda x: f"{x:.4f}")
    print(sens.sort_values(['metric','spearman_vs_original'], ascending=[True, False]).to_string(index=False))

    #####################
    # Plot the variations 
    #####################

    # Grid to evaluate the CR and BM on
    xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0 # Set min and max 

    grid_N = 200
    x = np.linspace(xmin, xmax, grid_N)
    y = np.linspace(ymin, ymax, grid_N)
    X, Y = np.meshgrid(x, y)

    # Set the colors to match the R colors
    custom_colors_bm = ["#E5E483", "#8ac926", "#4CCD99", "#5BBCFF", "#024CAA", "#00224D"]
    custom_colors_cr = ["#f5d7b0", "#f5be27", "#fa8d00", "#bc507f", "#7E1891", "#1C0159"]
    cmap_bm = LinearSegmentedColormap.from_list("bm_cmap", custom_colors_bm, N=256)
    cmap_cr = LinearSegmentedColormap.from_list("cr_cmap", custom_colors_cr, N=256)

    # Different variants to visualize for CR
    variants_CR = [
        ("CR scale=max-abs only (k=2, 0/45/90°)",          dict(scale='maxabs', k=2)),
        ("CR k=1 (z-score, max-abs, 0/45/90°)",            dict(scale='zscore_maxabs', k=1)),
        ("CR k=4 (z-score, max-abs, 0/45/90°)",            dict(scale='zscore_maxabs', k=4)),
        ("CR k=6 (z-score, max-abs, 0/45/90°)",            dict(scale='zscore_maxabs', k=6)),
        ("CR thetas=0,22.5,45 (z-score, max-abs, k=2)",    dict(scale='zscore_maxabs', k=2,
                                                            theta_flat=np.deg2rad(0),
                                                            theta_diag=np.deg2rad(22.5),
                                                            theta_perp=np.deg2rad(45))),
        ("CR thetas=-15,45,105 (z-score, max-abs, k=2)",   dict(scale='zscore_maxabs', k=2,
                                                            theta_flat=np.deg2rad(-15),
                                                            theta_diag=np.deg2rad(45),
                                                            theta_perp=np.deg2rad(105))),
    ]

    # For CR plots make a subplot grid big enough for all variants
    n = len(variants_CR)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.6*nrows), constrained_layout=True)
    axs = np.array(axs).reshape(-1) 

    # Run functions and plot each variant
    mappables = []
    for i, (title, params) in enumerate(variants_CR):
        ax = axs[i]
        min_, max_ = cr_bounds[i]
        cognitive_reserve = compute_CR(X, Y, cr_max=max_, cr_min=min_, **params)
        plt.sca(ax)
        m, _ = plot_panel(ax, X, Y, cognitive_reserve, title, "Cognitive Reserve", cmap=cmap_cr)
        mappables.append(m)

    cbar = fig.colorbar(mappables[-1], ax=axs[:len(variants_CR)], shrink=0.9, location='right', pad=0.02)
    cbar.set_label('Cognitive Reserve')

    # Now plot BM variants
    variants_BM = [
        ("BM scale=max-abs only (α=0.5, Euclidean)",         dict(scale='maxabs',       alpha_point=0.5, metric='euclid')),
        ("BM α=0.3 (z-score, max-abs, Euclidean)",           dict(scale='zscore_maxabs', alpha_point=0.3, metric='euclid')),
        ("BM α=0.7 (z-score, max-abs, Euclidean)",           dict(scale='zscore_maxabs', alpha_point=0.7, metric='euclid')),
        ("BM metric=Manhattan (α=0.5)",                      dict(scale='zscore_maxabs', alpha_point=0.5, metric='manhattan')),
        ("BM metric=Chebyshev (α=0.5)",                      dict(scale='zscore_maxabs', alpha_point=0.5, metric='cheby')),
        ("BM data-based",                                    {"_data_based": True})
    ]

    n_bm = len(variants_BM)
    ncols_bm = 3
    nrows_bm = int(np.ceil(n_bm / ncols_bm))
    fig_bm, axs_bm = plt.subplots(nrows_bm, ncols_bm, figsize=(4.8*ncols_bm, 4.6*nrows_bm), constrained_layout=True)
    axs_bm = np.array(axs_bm).reshape(-1)

    mappables_bm = []
    for i, (title, params) in enumerate(variants_BM):
        ax = axs_bm[i]
        plt.sca(ax)
        if params is None or params.get("_data_based", False):
            Z = compute_BM_data_based(X, Y)
        else:
            Z = compute_BM(X, Y, bm_max=bm_maxes[i], **params)
        m, _ = plot_panel(ax, X, Y, Z, title, "Brain Maintenance", cmap=cmap_bm)
        mappables_bm.append(m)

    cbar_bm = fig_bm.colorbar(mappables_bm[-1], ax=axs_bm[:n_bm], shrink=0.9, location='right', pad=0.02)
    cbar_bm.set_label('Brain Maintenance')

    plt.show()

if __name__ == "__main__":
    main()