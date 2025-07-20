#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

mouse = '/user/work/pr21872/SpaceData/regionsSampleData/MouseSupertypesamples43Regions.npy'
macaque = '/user/work/pr21872/SpaceData/regionsSampleData/float32samples141Regions.npy'
# Macaque (579356028,) found 8m values for reconstruction. Both 100k quantiles
# Mouse   (107678880,) found 1m values for reconsturction. Both 100k quantiles
samples = np.load(macaque)
samples_mouse = np.load(mouse)
print("Number of zeros macaque:", np.count_nonzero(samples == 0)/samples.size)
print("Number of zeros mouse:", np.count_nonzero(samples_mouse == 0)/samples_mouse.size)

"""NANS"""
# Count the number of NaN values per sample for samples
nan_counts_samples = np.sum(np.isnan(samples), axis=1)
mouse_nan_counts_samples = np.sum(np.isnan(samples_mouse), axis=1)
# Create bins so that each integer count is centered nicely:
bins_samples = np.arange(nan_counts_samples.min(), nan_counts_samples.max() + 2) - 0.5
# Create figure and 1x2 subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# --- First subplot: Macaque ---
counts_samples, bins_samples, _ = axs[0].hist(
    nan_counts_samples,
    bins=141,
    edgecolor='black'
)
axs[0].set_title('Macaque: NaNs per Sample (141 Regions)', fontsize=16)
axs[0].set_xlabel('# NaN Positions in a Sample', fontsize=16)
axs[0].set_ylabel('Frequency', fontsize=16)

# Tidy x-axis ticks for Macaque
bin_centers_samples = (bins_samples[:-1] + bins_samples[1:]) / 2
step1 = max(1, len(bin_centers_samples) // 10)
tick_positions1 = bin_centers_samples[::step1]
axs[0].set_xticks(tick_positions1.astype(int))
axs[0].tick_params(axis='x', rotation=45, labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].yaxis.get_offset_text().set_fontsize(14)

# --- Second subplot: Mouse ---
counts_mouse, bins_mouse, _ = axs[1].hist(
    mouse_nan_counts_samples,
    bins=43,
    edgecolor='black'
)
axs[1].set_title('Mouse: NaNs per Sample (43 Regions)', fontsize=16)
axs[1].set_xlabel('# NaN Positions in a Sample', fontsize=16)

# Tidy x-axis ticks for Mouse
bin_centers_mouse = (bins_mouse[:-1] + bins_mouse[1:]) / 2
step2 = max(1, len(bin_centers_mouse) // 10)
tick_positions2 = bin_centers_mouse[::step2]
axs[1].set_xticks(tick_positions2.astype(int))
axs[1].tick_params(axis='x', rotation=45, labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)

# Final layout
plt.tight_layout()
plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/NaN_distirbutions.png',dpi=600)

"""ALL (No NaNs)"""
import matplotlib.pyplot as plt
import numpy as np

# Flatten the arrays to 1D for histogramming
samples_flat = samples.flatten()
samples_mouse_flat = samples_mouse.flatten()

# Create figure with 1x2 subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# --- First subplot: Macaque ---
axs[0].hist(
    samples_flat,
    bins=100,
    edgecolor='black'
)
axs[0].set_title('Macaque: Distribution of Values', fontsize=14)
axs[0].set_xlabel('Average Gene Expression Value', fontsize=12)
axs[0].set_ylabel('Frequency', fontsize=12)
axs[0].tick_params(axis='x', labelsize=10, rotation=45)
axs[0].tick_params(axis='y', labelsize=10)

# --- Second subplot: Mouse ---
axs[1].hist(
    samples_mouse_flat,
    bins=100,
    edgecolor='black'
)
axs[1].set_title('Mouse: Distribution of Values', fontsize=14)
axs[1].set_xlabel('Average Gene Expression Value', fontsize=12)
axs[1].set_ylabel('Frequency', fontsize=12)
axs[1].tick_params(axis='x', labelsize=10, rotation=45)
axs[1].tick_params(axis='y', labelsize=10)

# Final layout
plt.tight_layout()
plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/all_dist.png',dpi=600)

# Set global font size
# plt.rcParams.update({'font.size': 14})  # <<== Increase all fonts globally

# """Original"""
# samples_flat = samples.reshape(-1, 1)

# plt.hist(samples_flat, bins=100, alpha=0.5, label='QT Inverse', density=True)
# plt.legend()
# plt.title("Distribution Comparison: Original vs QT Inverse")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.tight_layout()

# plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/macaque_141.png',dpi=600)

"""Clipping then QT"""
# # 1) flatten & drop NaNs from full data
# full_flat = samples.reshape(-1, 1)   
# full_flat = full_flat[~np.isnan(full_flat)]

# # 2) compute clip threshold
# upper_q = np.percentile(full_flat, 90)

# # 3) clip full_flat
# clipped_full = np.minimum(full_flat, upper_q)

# # 4) reshape to 2D for QT
# clipped_full = clipped_full.reshape(-1, 1)

# # 5) fit QT
# qt = QuantileTransformer(
#     output_distribution='uniform',
#     random_state=42,
#     n_quantiles=int(1e5),
#     subsample=int(1e6)
# )
# qt.fit(clipped_full)

# # 6) now apply to your mapping subset:
# #    (flatten, clip, reshape, transform, then reshape back if you like)
# mapping_clipped  = np.minimum(full_flat, upper_q)
# mapping_clipped  = mapping_clipped.reshape(-1, 1)
# mapping_qt_flat  = qt.transform(mapping_clipped).flatten()

# # 7) plot the histogram
# plt.figure()
# plt.hist(mapping_qt_flat, bins=100, alpha=0.5, label='Clip+QT', density=True)
# plt.legend()
# plt.title("Clip + QuantileTransformer")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.tight_layout()
# plt.savefig('/user/work/pr21872/SpaceData/regionsSampleData/macaque_clip_QT_90.png', dpi=600)

"""logminmax clip"""
# all_values = samples.flatten()
# qt = QuantileTransformer(
#     output_distribution='uniform',
#     random_state=42,
#     n_quantiles=int(1e4),
#     subsample=int(1e7) 
# )
# X_transformed = qt.fit_transform(all_values.reshape(-1, 1)).flatten()
# plt.figure(figsize=(12, 6))
# # plt.subplot(2,2,3)
# plt.subplot(1,2,1)
# # Flatten the samples array and remove NaN values
# all_values = samples.flatten()
# all_values_no_nans = all_values[~np.isnan(all_values)]
# all_values_no_nans_no_0s = all_values_no_nans[all_values_no_nans!=0.0]
# plt.hist(all_values_no_nans, bins=100, edgecolor='black')  # adjust bins as needed
# plt.xlim(0)
# plt.xlabel('Values from sequences')
# plt.ylabel('Frequency')
# plt.title('Original')

# plt.subplot(1,2,2)
# # Flatten the samples array and remove NaN values
# # Apply log scaling
# log_all_values = np.log(all_values_no_nans+1)
# min_val = log_all_values.min()
# max_val = log_all_values.max()
# # Apply min–max scaling on all valid values, preserving NaNs.
# scaled_all_values = (log_all_values - min_val) / (max_val - min_val)
# plt.hist(X_transformed, bins=100, edgecolor='black')  # adjust bins as needed
# plt.xlim(0, 1)
# plt.xlabel('Values from sequences after QT transform')
# plt.ylabel('Frequency')
# plt.title('Uniform QT Transform')

# plt.tight_layout()
# # plt.suptitle('Mouse Samples Distributions (NaNs not included)', fontsize=16, y=1.02)
# plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/mouse_QT_transform.png', dpi=600)

"""Power transformer"""
# from sklearn.preprocessing import PowerTransformer

# # BC requires strictly positive x; YJ handles zeros too.
# pt = PowerTransformer(method='yeo-johnson', standardize=False)
# pt.fit(samples_flat)
# mapped = pt.transform(samples_flat)

# # plt.hist(all_values.flatten(), bins=100, alpha=0.5, label='Original', density=True)
# plt.hist(mapped, bins=100, alpha=0.5, label='QT Inverse', density=True)
# plt.legend()
# plt.title("Distribution Comparison: Original vs QT Inverse")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.tight_layout()

# plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/macaque_transform_200k_quantile.png',dpi=600)


"""Robust Scalar shape"""
# samples_flat = samples.flatten().reshape(-1, 1)  # shape: (1000000*43, 1)

# # Plot histogram of the original distribution
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.hist(samples_flat, bins=50, color='skyblue', edgecolor='k')
# plt.title("Original Data Distribution")
# plt.xlabel("Value")
# plt.ylabel("Frequency")

# # Apply Robust Scaling
# scaler = RobustScaler()
# samples_scaled = scaler.fit_transform(samples_flat)

# # Plot histogram of the robust-scaled distribution
# plt.subplot(1, 2, 2)
# plt.hist(samples_scaled, bins=50, color='salmon', edgecolor='k')
# plt.title("After Robust Scaling")
# plt.xlabel("Scaled Value")
# plt.ylabel("Frequency")

# plt.tight_layout()
# plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/macaque_robust_scaling.png',dpi=600)

"""Percentage NaNs"""
# non_nan_or_nonzero_row_indices = np.where(
#     np.all(~np.isnan(samples), axis=1)
# )[0]
# print("Indices of rows with all positions that are not NaN:", non_nan_or_nonzero_row_indices)
# print('Total number samples with all positions non NaN:')
# print(len(non_nan_or_nonzero_row_indices))
# print('Total samples')
# print(len(samples))

# print()
# print('Macaque % of samples that have no NaNs:')
# print(len(non_nan_or_nonzero_row_indices)/len(samples))

# total_entries = samples.size
# nan_entries   = np.isnan(samples).sum()
# nan_pct       = nan_entries / total_entries * 100

# print(f"Overall NaN entries: {nan_entries:,} / {total_entries:,}")
# print(f"Overall percentage of NaNs: {nan_pct:.2f}%")



"""Testing transform"""
# from sklearn.metrics import mean_squared_error, r2_score
# # Initialize the QuantileTransformer
# qt = QuantileTransformer(
#     output_distribution='uniform',
#     random_state=42,
#     n_quantiles=int(1e4),
#     subsample=int(1e7) 
# )
# X_transformed = qt.fit_transform(all_values.reshape(-1, 1)).flatten()

# all_values = samples.flatten()
# all_values = all_values[~np.isnan(all_values)]
# # Apply transform and inverse transform
# X_transformed = qt.fit_transform(all_values.reshape(-1, 1)).flatten()
# # X_reconstructed = qt.inverse_transform(X_transformed.reshape(-1, 1)).flatten()

# # # Calculate metrics
# # all_values = all_values[~np.isnan(all_values)]
# # mse = mean_squared_error(all_values, X_reconstructed)
# # r2 = r2_score(all_values, X_reconstructed)

# # print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

# plt.hist(all_values.flatten(), bins=100, alpha=0.5, label='Original', density=True)
# plt.hist(X_transformed.flatten(), bins=100, alpha=0.5, label='QT Inverse', density=True)
# plt.legend()
# plt.title("Distribution Comparison: Original vs QT Inverse")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.tight_layout()

# plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/macaque_transform_200k_quantile.png',dpi=600)


"""Plotting Distributions"""
# fig_name = 'mouse_no_0s'

# # Count the number of 0.0 values per sample for samples43
# zero_counts_samples = np.sum(samples==0.0, axis=1)
# bins_samples = np.arange(zero_counts_samples.min(), zero_counts_samples.max() + 2) - 0.5

# # Count the number of NaN values per sample for samples43
# NaN_counts_samples = np.sum(np.isnan(samples), axis=1)
# NaN_bins_samples = np.arange(NaN_counts_samples.min(), NaN_counts_samples.max() + 2) - 0.5


# plt.figure(figsize=(12, 10))  # Set your desired width and height here
# plt.subplot(2,2,1)

# counts_samples43, bins_samples, patches_samples43 = plt.hist(zero_counts_samples, bins=bins_samples, edgecolor='black')
# plt.xlabel('Number of 0.0 values per sample')
# plt.ylabel('Frequency')
# plt.title('Distribution of 0.0 counts per sample ')

# # Calculate bin centers and select tick positions for samples43
# bin_centers_samples43 = (bins_samples[:-1] + bins_samples[1:]) / 2
# plt.xticks(bin_centers_samples43[::2])  # every second tick
# # zero_tick_values_samples43 = bin_centers_samples43[counts_samples43 > 3000]
# # plt.xticks(zero_tick_values_samples43)


# plt.subplot(2,2,2)

# counts_samples43, bins_samples, patches_samples43 = plt.hist(NaN_counts_samples, bins=NaN_bins_samples, edgecolor='black')
# plt.xlabel('Number of NaN values per sample')
# plt.ylabel('Frequency')
# plt.title('Distribution of NaN counts per sample ')

# # Calculate bin centers and select tick positions for samples43
# bin_centers_samples43 = (bins_samples[:-1] + bins_samples[1:]) / 2
# plt.xticks(bin_centers_samples43[::2])  # every second tick
# # NaN_tick_values_samples43 = bin_centers_samples43[counts_samples43 > 30000]
# # plt.xticks(NaN_tick_values_samples43)


# plt.subplot(2,2,3)
# all_values = samples.flatten()
# all_values_no_nans = all_values[~np.isnan(all_values)]
# all_values_no_nans_no_0s = all_values_no_nans[all_values_no_nans!=0.0]
# plt.hist(all_values_no_nans, bins=100, edgecolor='black')  # adjust bins as needed
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of all values in samples (NaN not included)')

# plt.subplot(2,2,4)
# # Apply log scaling
# log_all_values = np.log(all_values_no_nans+1)
# min_val = log_all_values.min()
# max_val = log_all_values.max()
# # Apply min–max scaling on all valid values, preserving NaNs.
# scaled_all_values = (log_all_values - min_val) / (max_val - min_val)
# plt.hist(scaled_all_values, bins=100, edgecolor='black')  # adjust bins as needed
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of all values in samples (NaN not included)')


# plt.suptitle('Mouse Samples (with supertypes) distributions', fontsize=16, y=1.02)
# plt.tight_layout()
# plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/{fig_name}.png',dpi=600)


# plt.subplot(2,2,4)
# Initialize the QuantileTransformer
# qt = QuantileTransformer(
#     output_distribution='uniform',
#     random_state=42,
#     n_quantiles=int(1e5),
#     subsample=int(10e6) 
# )
# transformed_data = qt.fit_transform(all_values_no_nans.reshape(-1, 1)).flatten()


# Fit and transform the data
# transformed_data = qt.fit_transform(all_values_no_nans.reshape(-1, 1)).flatten()
# reversed_data = qt.inverse_transform(transformed_data.reshape(-1, 1)).flatten()

# set all nan values to -1.0
# all_values_nans = np.nan_to_num(all_values_nans, nan=-1.0)
# combined_data = np.concatenate(all_values_nans,transformed_data)



