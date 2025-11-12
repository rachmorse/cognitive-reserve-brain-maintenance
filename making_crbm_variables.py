import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# NOTE that this script should be run using the scaled residuals from an age-cognition change and an age-brain change regression.
# These residuals allow CR and BM to be adjusted for age and to account for the steeper brain and cognitive change slopes observed in older age.
# So, for example, in the paper we use X = scaled hippocampus change residuals, Y = scaled episodic memory change residuals
# The scaling in the paper uses z-scoring then scaling to [-1, 1] after taking the residuals.

def create_cr_measure(X, Y):
    """Create the cognitive reserve measure based on X and Y values.

    Args:
        X (np.ndarray): Array of scaled brain change residuals.
        Y (np.ndarray): Array of scaled change in cognition residuals.

    Returns:
        np.ndarray: Array of cognitive reserve values.
    """
    # Step 1. Define the angles for each quadrant
    theta_flat = np.deg2rad(0.0)        # top-left -> flat lines when cognitive reserve is high
    theta_diag = np.deg2rad(45.0)       # top-right & bottom-left -> diagonal for when cognitive reserve is medium
    theta_perp = np.deg2rad(90.0)       # bottom-right -> perpendicular lines when cognitive reserve is low

    # Step 2. Define k as a constant for how smooth the transitions between quadrants should be 
    # Higher k = sharper transitions, lower k = smoother transitions
    kx, ky = 2, 2

    # Step 3. Define wx and wy which determine where each point is on the grid
    # Hyperbolic tangent smoothly transitions between -1 and 1 along s-shaped curve
    wx = 0.5 * (1 + np.tanh(kx * X))
    wy = 0.5 * (1 + np.tanh(ky * Y))

    # Then use these to compute a blended orientation angle based on the quadrant
    # (1-wx)*(wy) ~ 1 at top-left, 0 at bottom-right
    # (wx)*(wy) ~ 1 at top-right, 0 at bottom-left
    # (1-wx)*(1-wy) ~ 1 at bottom-left, 0 at top-right
    # (wx)*(1-wy) ~ 1 at bottom-right, 0 at top-left
    # Summing all four of these will always equal 1

    # Step 4. Define theta(x,y) 
    theta = (1 - wx)*(wy)*theta_flat + \
            wx*(1 - wy)*theta_perp + \
            ((wx)*(wy) + (1 - wx)*(1 - wy))*theta_diag

    # Step 5. Compute the cognitive reserve value
    cognitive_reserve = np.cos(theta) * Y - np.sin(theta) * X

    # Step 6. Normalize 
    # Pre-calculated max / min possible value for this dataset with k and theta defined as above
    # If changing k or theta you will need to recalculate the bound which is the maximum possible 
    # cognitive reserve value, rather than the observed max in your data
    bound = 1.0484785942244885

    # This formula assumes symmetric bounds
    return (cognitive_reserve + bound) / (2 * bound)


def create_bm_measure(X, Y):
    """Create the brain maintenance measure based on X and Y values.

    Args:
        X (np.ndarray): Array of scaled brain change residuals.
        Y (np.ndarray): Array of scaled change in cognition residuals.

    Returns:
        np.ndarray: Array of brain maintenance values.
    """
    # Calculate distance to the brain maintenance point (1, 1) 
    dist_bm_point_raw = np.sqrt((X - 1)**2 + (Y - 1)**2)

    # Calculate distance to the brain maintenance line x = y
    dist_bm_line_raw  = np.abs(X - Y) / np.sqrt(2)

    # Combine both distances and reverse the value so that higher brain maintenance means better maintenance
    brain_maintenance = 3.414214 - (dist_bm_point_raw + dist_bm_line_raw)  # pre-calculated max brain maintenance value

    # Normalize and return
    return brain_maintenance / 3.414214 


def plot_crbm_measures(X, Y, cognitive_reserve, brain_maintenance):
    """Plot the cognitive reserve and brain maintenance measures on separate subplots.

    Args:
        X (np.ndarray): Array of scaled brain change residuals.
        Y (np.ndarray): Array of scaled change in cognition residuals.
        cognitive_reserve (np.ndarray): Array of cognitive reserve values.
        brain_maintenance (np.ndarray): Array of brain maintenance values.        
    """
    # Define custom colors - these are the ones used in the paper matching R colors
    custom_colors_bm = ["#E5E483", "#8ac926", "#4CCD99", "#5BBCFF", "#024CAA", "#00224D"]
    custom_colors_cr = ["#f5d7b0", "#f5be27", "#fa8d00", "#bc507f", "#7E1891", "#1C0159"]
    cmap_bm = LinearSegmentedColormap.from_list("bm_cmap", custom_colors_bm, N=256)
    cmap_cr = LinearSegmentedColormap.from_list("cr_cmap", custom_colors_cr, N=256)

    # Combine plots into one figure
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.65))  

    # Cognitive reserve plot
    plt.sca(axs[0])
    plt.xlabel("Brain Change (age-adjusted)")
    plt.ylabel("Change in Cognition (age-adjusted)")
    mesh = plt.pcolormesh(X, Y, cognitive_reserve, shading='auto', cmap=cmap_cr)
    levels = np.linspace(0, 1, 15)
    contours = plt.contour(X, Y, cognitive_reserve, levels=levels, colors='black', linewidths=0.6, alpha=1)
    plt.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    plt.axhline(0, color='white', linewidth=1.0, alpha=0.8)
    plt.axvline(0, color='white', linewidth=1.0, alpha=0.8)
    plt.colorbar(mesh, ax=axs[0], label='Cognitive Reserve')

    # Brain maintenance plot
    plt.sca(axs[1])
    plt.xlabel("Brain Change (age-adjusted)")
    plt.ylabel("Change in Cognition (age-adjusted)")
    mesh = plt.pcolormesh(X, Y, brain_maintenance, shading='auto', cmap=cmap_bm)
    levels = np.linspace(brain_maintenance.min(), brain_maintenance.max(), 15)
    contours = plt.contour(X, Y, brain_maintenance, levels=levels, colors='black', linewidths=0.6, alpha=1)
    plt.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    plt.axhline(0, color='white', linewidth=1.0, alpha=0.8)
    plt.axvline(0, color='white', linewidth=1.0, alpha=0.8)
    plt.colorbar(mesh, ax=axs[1], label='Brain Maintenance')

    plt.tight_layout()
    plt.show()


def main():
    # If you want to run this code on simulated data, optionally create a dataset with 800x800 points, range [-1, 1] 
    # Otherwise, load your own scaled brain and cognition change residuals here 
    x = np.linspace(-1, 1, 800)
    y = np.linspace(-1, 1, 800)
    X, Y = np.meshgrid(x, y)  # X and Y are a 2D grid of coordinates (all possible combinations of x and y)

    # Create cognitive reserve and brain maintenance measures
    cognitive_reserve = create_cr_measure(X, Y)
    brain_maintenance = create_bm_measure(X, Y)

    # Optionally choose to plot the simulated cognitive reserve and brain maintenance measures
    plot_crbm_measures(X, Y, cognitive_reserve, brain_maintenance)


if __name__ == "__main__":
    main()