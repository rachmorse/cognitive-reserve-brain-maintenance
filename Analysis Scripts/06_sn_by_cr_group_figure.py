import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

from nilearn import datasets, surface, plotting

def plot_salience_network(roi_dict, min_val, max_val, atlas_path, output_file):
    """
    Plot a multi-view brain surface image using an ROI dictionary for 
    the Shirer atlas with weighted connectivity values.

    Args:
        roi_dict (dict): Dictionary mapping ROI names to connectivity values (e.g. where the average connectivty values are for each ROI).
        min_val (float): Minimum value for color scaling.
        max_val (float): Maximum value for color scaling.
        atlas_path (str): Path to the Shirer atlas NIfTI file.
        output_file (str): Path to save the output image.
    """
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata() 
    affine = atlas_img.affine

    # ROI-volume index map according to the Shirer atlas
    roi_index_map = {
        "AS_L_midFront": 0,
        "AS_L_ins": 1,
        "AS_RL_acc_medPref_sma": 2,
        "AS_R_midFront": 3,
        "AS_R_ins": 4,
        "AS_L_lobule_VI_crus_I": 5,
        "AS_R_lobule_VI_crus_I": 6
    }

    # Create array filled with NaN so that by default all voxels are 'greyed out'
    weighted_map = np.full(atlas_data.shape[:3], np.nan, dtype=np.float32)

    # Assign values from roi_dict onto the map
    for roi_name, val in roi_dict.items():
        if roi_name in roi_index_map:
            idx = roi_index_map[roi_name]

            roi_mask = atlas_data[..., idx]
            # Set locations corresponding to `roi_mask == 1` to val
            mask_indices = roi_mask > 0
            weighted_map[mask_indices] = val

    weighted_img = nib.Nifti1Image(weighted_map, affine)

    # Project volume to surfaces
    fsaverage = datasets.fetch_surf_fsaverage()
    texture_left = surface.vol_to_surf(weighted_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(weighted_img, fsaverage.pial_right)

    # Set color limits for the surface plots
    vmin, vmax = min_val, max_val

    # Create a temporary folder for each view of the surface
    temp_dir = "temp_surf_outputs"
    os.makedirs(temp_dir, exist_ok=True)

    views = [
        ("left", "lateral"),
        ("left", "medial"),
        ("right", "medial"),
        ("right", "lateral")
    ]
    img_paths = []

    # Plot the 4 views
    for hemi, view in views:
        filename = f"{hemi}_{view}.png"
        out_path = os.path.join(temp_dir, filename)
        plotting.plot_surf_stat_map(
            fsaverage.pial_left if hemi == "left" else fsaverage.pial_right,
            texture_left if hemi == "left" else texture_right,
            hemi=hemi,
            view=view,
            colorbar=False,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            threshold=None,
            bg_map=fsaverage.sulc_left if hemi == "left" else fsaverage.sulc_right,
            bg_on_data=True,
            title=None,
            output_file=out_path
        )
        img_paths.append(out_path)
        
    # Create a separate color bar
    fig_cbar, ax_cbar = plt.subplots(figsize=(0.05, 0.4)) # Needed to adjust size to get it to fit with the other images
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar, format="%.2f")
    cbar.ax.tick_params(labelsize=2, length=1.5, width=0.2)
    cbar.outline.set_linewidth(0.2) 
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    cbar_out_path = os.path.join(temp_dir, "colorbar.png")
    fig_cbar.savefig(cbar_out_path, bbox_inches="tight", dpi=1000)
    plt.close(fig_cbar)

    # Combine the four snapshots + color bar in one figure
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    # Four brain views
    for i, path in enumerate(img_paths):
        img = plt.imread(path)
        axes[i].imshow(img)
        axes[i].axis("off")

    # Separate color bar
    cbar_img = plt.imread(cbar_out_path)
    axes[-1].imshow(cbar_img)
    axes[-1].axis("off")

    plt.subplots_adjust(wspace=-0.07, hspace=0)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Clean up temp snapshots
    for path in img_paths:
        os.remove(path)
    os.remove(cbar_out_path)
    os.rmdir(temp_dir)

    print(f"Multi-view surface image saved to {output_file}")

if __name__ == "__main__":
    low_cr = {
        'AS_L_Ins': [-0.143247504710141],
        'AS_L_lobule_VI_crus_I': [-0.012034497346713],
        'AS_L_midFront': [-0.101147051229879],
        'AS_RL_acc_medPref_sma': [-0.205194477014672],
        'AS_R_Ins': [-0.131497558838432],
        'AS_R_lobule_VI_crus_I': [-0.096921386482409],
        'AS_R_midFront': [-0.131256527672683]
    }

    high_cr = {
        'AS_L_Ins': [0.001876820015408],
        'AS_L_lobule_VI_crus_I': [0.011015563282694],
        'AS_L_midFront': [0.014845294973636],
        'AS_RL_acc_medPref_sma': [0.105486184459808],
        'AS_R_Ins': [0.066456359593062],
        'AS_R_lobule_VI_crus_I': [0.047532963175777],
        'AS_R_midFront': [0.075332885362378]
    }

    # Calculate the difference between high and low CR groups
    difference_cr = {}
    for roi in high_cr.keys():
        difference_cr[roi] = high_cr[roi][0] - low_cr[roi][0]
    diff_values = list(difference_cr.values())

    # Set atlas path
    atlas_path = "/Users/rachelmorse/Documents/2023:2024/CR & BM Project/Task invariant network/subrois_shirer2012_mod_4d.nii.gz"
    
    plot_salience_network(
        difference_cr,
        0,
        0.2,
        atlas_path,
        output_file="high_vs_low_cr_difference_surface.png"
        )