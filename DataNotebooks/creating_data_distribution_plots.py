#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import QuantileTransformer

mouse = '/user/work/pr21872/SpaceData/regionsSampleData/MouseSupertypesamples43Regions.npy'
macaque = '/user/work/pr21872/SpaceData/regionsSampleData/float32samples141Regions.npy'
# Macaque (579356028,) found 8m values for reconstruction. Both 100k quantiles
# Mouse   (107678880,) found 1m values for reconsturction. Both 100k quantiles
samples = np.load(macaque)

"""Printing shape"""
print(samples.shape)

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

"""Testing transform"""
# from sklearn.metrics import mean_squared_error, r2_score
# # Initialize the QuantileTransformer
# qt = QuantileTransformer(
#     output_distribution='uniform',
#     random_state=42,
#     n_quantiles=int(1e6),
#     subsample=int(10e6) 
# )

# all_values = samples.flatten()
# all_values = all_values[~np.isnan(all_values)]
# # Apply transform and inverse transform
# X_transformed = qt.fit_transform(all_values.reshape(-1, 1)).flatten()
# X_reconstructed = qt.inverse_transform(X_transformed.reshape(-1, 1)).flatten()

# # Calculate metrics
# # all_values = all_values[~np.isnan(all_values)]
# # mse = mean_squared_error(all_values, X_reconstructed)
# # r2 = r2_score(all_values, X_reconstructed)

# # print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

# plt.hist(all_values.flatten(), bins=100, alpha=0.5, label='Original', density=True)
# plt.hist(X_reconstructed.flatten(), bins=100, alpha=0.5, label='QT Inverse', density=True)
# plt.legend()
# plt.title("Distribution Comparison: Original vs QT Inverse")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.tight_layout()

# plt.savefig(f'/user/work/pr21872/SpaceData/regionsSampleData/macaque_transform_test.png',dpi=600)


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



