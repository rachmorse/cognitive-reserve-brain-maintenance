import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import statsmodels.formula.api as smf
from matplotlib.collections import LineCollection
from numpy import arctanh, sqrt
from scipy.stats import spearmanr

#####################################
# Create CRBM Variables
#####################################
data = pd.read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Clean Data All Cohorts/Wide Format/Analysed data.csv")
X_raw = pd.to_numeric(data["res_hc_slopes_age"], errors="coerce")   # memory residuals
Y_raw = pd.to_numeric(data["res_mem_slopes_age"],  errors="coerce")   # hippocampal residuals

# Drop NA
m = X_raw.notna() & Y_raw.notna()
X_raw, Y_raw = X_raw[m], Y_raw[m]

# Z-score normalization
X_raw = (X_raw - X_raw.mean()) / X_raw.std()
Y_raw = (Y_raw - Y_raw.mean()) / Y_raw.std()

# Normalize to [-1, 1]
X_raw = X_raw / max(abs(X_raw.min(skipna=True)), abs(X_raw.max(skipna=True)))
Y_raw = Y_raw / max(abs(Y_raw.min(skipna=True)), abs(Y_raw.max(skipna=True)))

# Calculate CR values
theta_flat = np.deg2rad(0.0)     
theta_diag = np.deg2rad(45.0)       
theta_perp = np.deg2rad(90.0)     
kx, ky = 2, 2
wX_raw = 0.5 * (1 + np.tanh(kx * X_raw))
wY_raw = 0.5 * (1 + np.tanh(ky * Y_raw))
theta_raw = ((1 - wX_raw) * (wY_raw) * theta_flat +
             wX_raw * (1 - wY_raw) * theta_perp +
             ((wX_raw * wY_raw) + (1 - wX_raw) * (1 - wY_raw)) * theta_diag)
CR_raw = np.cos(theta_raw) * Y_raw - np.sin(theta_raw) * X_raw

# Then normalize accordingly
CR_bound = 1.0484785942244885 # pre-calculated max / min possible value for this dataset
CR_raw = (CR_raw + CR_bound) / (2 * CR_bound)

# Calculate BM values
dist_bm_point_raw = np.sqrt((X_raw - 1)**2 + (Y_raw - 1)**2)
dist_bm_line_raw  = np.abs(X_raw - Y_raw) / np.sqrt(2)
BM_raw = 3.414214 - (dist_bm_point_raw + dist_bm_line_raw) # pre-calculated max BM value
BM_raw = BM_raw / 3.414214 # Scale to 0-1

# Define custom colors to match R
custom_colors_cr = ["#f5d7b0", "#f5be27", "#fa8d00", "#bc507f", "#7E1891", "#1C0159"]
cmap_cr = LinearSegmentedColormap.from_list("cr_cmap", custom_colors_cr, N=256)

# Add the new vars back into the df
id_col = "id"                      
ids = data.loc[m, id_col].reset_index(drop=True)

df_raw = pd.DataFrame({
    id_col: ids,
    "X": X_raw.reset_index(drop=True),
    "Y": Y_raw.reset_index(drop=True),
    "CR":    CR_raw.reset_index(drop=True),
    "BM":    BM_raw.reset_index(drop=True),
})
df_raw = df_raw.sort_values('CR').reset_index(drop=True)

# Split CR into bins
n_bins = 70
df_raw['CR_bin'] = pd.qcut(df_raw['CR'], q=n_bins, labels=False)
records = []
x_line = np.linspace(df_raw['X'].min(), df_raw['X'].max(), 100)

# Make line segments for each bin with >=5 points
for b in range(n_bins):
    grp = df_raw[df_raw['CR_bin'] == b]
    if len(grp) < 5:
        continue
    a, c = np.polyfit(grp['X'], grp['Y'], 1)
    records.append((grp['CR'].mean(), a, c))
lines = pd.DataFrame(records, columns=['CR_mean','slope','intercept'])

# Build line segments
segments = []
colors = []
for _, row in lines.iterrows():
    y_line = row['slope'] * x_line + row['intercept']
    points = np.array([x_line, y_line]).T.reshape(-1,1,2)
    segments.append(points)
    colors.append(row['CR_mean'])

# Create the line collection
norm = plt.Normalize(vmin=0, vmax=1)
lc = LineCollection([seg[:,0,:] for seg in segments],
                    cmap=cmap_cr, norm=norm,
                    linewidths=5, alpha=0.9)
lc.set_array(np.array(colors))

# Plot the way CR moderates the x-y relationship
fig, ax = plt.subplots(figsize=(8,6))
ax.add_collection(lc)
ax.autoscale()
ax.set_xlabel("Hippocampal Annual Change (age-adjusted)")
ax.set_ylabel("Memory Annual Change (age-adjusted)")
ax.set_title("Change in the Memory–Hippocampal Relationship Across Cognitive Reserve")
cb = plt.colorbar(lc, ax=ax, label='Cognitive Reserve')
cb.ax.tick_params(labelsize=9)
ax.grid(alpha=0.3)
ax.set_facecolor('white')
plt.tight_layout()
plt.show()

df_raw.to_csv("/tsd/p274/data/durable/projects/p027-cr_bm/crbm_data.csv", index=False)