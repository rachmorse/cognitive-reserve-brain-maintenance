import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numpy import sqrt

# NOTE that this script should be run using the scaled residuals from an age-memory and an age-hippocampal volume regression.
# So X = scaled hippocampal residuals, Y = scaled memory residuals
# The scaling used in the paper is to z-score then scale to [-1, 1] after taking the residuals.

# If you want to run this code on simulated data, optionally create a dataset with 800x800 points, range [-1, 1] 
x = np.linspace(-1, 1, 800)
y = np.linspace(-1, 1, 800)
X, Y = np.meshgrid(x, y) # X and Y are a 2D grid of coordinates (all possible combinations of x and y)

#######################
# Create the CR metric
#######################

# Step 1. Define the angles for each quadrant
theta_flat = np.deg2rad(0.0)        # top-left -> flat lines when CR is high
theta_diag = np.deg2rad(45.0)       # top-right & bottom-left -> diagonal for when CR is medium
theta_perp = np.deg2rad(90.0)       # bottom-right -> perpendicular lines when CR is low

# Step 2. Define k as a constant for how smooth the transitions between quadrants should be 
# Higher k = sharper transitions, lower k = smoother transitions
kx, ky = 2, 2

# Step 3. Define wx and wy which determine where each point is on the grid
# Hyperbolic tangent smoothly transitions between -1 and 1 along s-shaped curve
wx = 0.5 * (1 + np.tanh(kx * X))
wy = 0.5 * (1 + np.tanh(ky * Y))

# Then use these to compute a blended orientation angle based on the quadrant
# Top left = (1-wx)*(wy) ~ 1 at top-left, 0 at bottom-right
# Top right = (wx)*(wy) ~ 1 at top-right, 0 at bottom-left
# Bottom left = (1-wx)*(1-wy) ~ 1 at bottom-left, 0 at top-right
# Bottom right = (wx)*(1-wy) ~ 1 at bottom-right, 0 at top-left
# Summing all four of these will always equal 1

# Step 4. Define theta(x,y) as a smooth blend of the angles in each quadrant
theta = (1 - wx)*(wy)*theta_flat + \
        wx*(1 - wy)*theta_perp + \
        ((wx)*(wy) + (1 - wx)*(1 - wy))*theta_diag

# Step 5. Finally compute the cognitive reserve value
cognitive_reserve = np.cos(theta) * Y - np.sin(theta) * X

# Step 6. Normalize 
bound = 1.0484785942244885 # pre-calculated max / min possible value for this dataset
cognitive_reserve = (cognitive_reserve + bound) / (2 * bound) 

#######################
# Create the BM metric
#######################

# Calculate distance to the BM point (1, 1) 
dist_bm_point_raw = np.sqrt((X - 1)**2 + (Y - 1)**2)

# Calculate distance to the BM line x = y
dist_bm_line_raw  = np.abs(X - Y) / np.sqrt(2)

# Combine both distances and reverse the value so that higher BM means better maintenance
brain_maintenance = 3.414214 - (dist_bm_point_raw + dist_bm_line_raw) # pre-calculated max BM value

# Normalize
brain_maintenance = brain_maintenance / 3.414214 

####################
# Optional plotting 
####################

# Define custom colors - these are the ones used in the paper matching R colors
custom_colors_bm = ["#E5E483", "#8ac926", "#4CCD99", "#5BBCFF", "#024CAA", "#00224D"]
custom_colors_cr = ["#f5d7b0", "#f5be27", "#fa8d00", "#bc507f", "#7E1891", "#1C0159"]
cmap_bm = LinearSegmentedColormap.from_list("bm_cmap", custom_colors_bm, N=256)
cmap_cr = LinearSegmentedColormap.from_list("cr_cmap", custom_colors_cr, N=256)

# Combine plots into one figure
fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.65))  

# CR plot
plt.sca(axs[0])
plt.xlabel("Hippocampal Annual Change (age-adjusted)")
plt.ylabel("Memory Annual Change (age-adjusted)")
mesh = plt.pcolormesh(X, Y, cognitive_reserve, shading='auto', cmap=cmap_cr)
levels = np.linspace(0, 1, 15)
contours = plt.contour(X, Y, cognitive_reserve, levels=levels, colors='black', linewidths=0.6, alpha=1)
plt.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
plt.axhline(0, color='white', linewidth=1.0, alpha=0.8)
plt.axvline(0, color='white', linewidth=1.0, alpha=0.8)
plt.colorbar(mesh, ax=axs[0], label='Cognitive Reserve')

# BM plot
plt.sca(axs[1])
plt.xlabel("Hippocampal Annual Change (age-adjusted)")
plt.ylabel("Memory Annual Change (age-adjusted)")
mesh = plt.pcolormesh(X, Y, brain_maintenance, shading='auto', cmap=cmap_bm)
levels = np.linspace(brain_maintenance.min(), brain_maintenance.max(), 15)
contours = plt.contour(X, Y, brain_maintenance, levels=levels, colors='black', linewidths=0.6, alpha=1)
plt.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
plt.axhline(0, color='white', linewidth=1.0, alpha=0.8)
plt.axvline(0, color='white', linewidth=1.0, alpha=0.8)
plt.colorbar(mesh, ax=axs[1], label='Brain Maintenance')

plt.tight_layout()
plt.show()