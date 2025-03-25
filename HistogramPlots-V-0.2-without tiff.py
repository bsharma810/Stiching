
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')

# Define main root directory where all exp-time subfolders are present
main_root_dir = r"C:\Users\syagi\Documents\PH007\20X\CH3"    # change as per your folder name

# Define a single PDF filename to store all histogram plots and images
pdf_filepath = os.path.join(main_root_dir, "PH007-20X-CH3.pdf") # Name of the Pdf file in double quotes " " 
                                                               # Assign the name same as folder name

# Make a list of custom colors
custom_colors = [
    (0/255, 0/255, 0/255),      # Black (Darkest intensity)
    (255/255, 0/255, 0/255),    # Red
    (0/255, 255/255, 0/255),    # Green
    (179/255, 140/255, 0/255)   # Mustard 
]

# Function to create a segmented image using the LUT
def create_segmented_image(assigned_components, sorted_colors):
    assigned_components = assigned_components.astype(int)
    
    # Create an empty image
    segmented_image = np.zeros((*assigned_components.shape, 3), dtype=np.float32)
   
    # Assign colors based on GMM clustering
    for i in range(len(sorted_colors)):  
        mask = (assigned_components == i)  # Assign colors based on cluster index
        if np.any(mask):  
            segmented_image[mask] = sorted_colors[i]     

    # Assign color to pixels at 65535
    magenta_mask =  (img_array == 65535)
    segmented_image[magenta_mask] = (255/255, 0/255, 255/255)       
    return segmented_image


# Open PDF for saving
with PdfPages(pdf_filepath) as pdf:
    print(f"Saving all histograms with Gaussian fits and images in {pdf_filepath}")

    # First Loop through each exposure-time folder
    for exposure_folder in sorted(os.listdir(main_root_dir)):  # Sort for consistency
        exposure_path = os.path.join(main_root_dir, exposure_folder)
        if not os.path.isdir(exposure_path):
            continue

        print(f"Processing exposure folder: {exposure_folder}")
        histograms = []
        tiff_images = []

        # Second Loop through subfolders (XY01, XY02, etc.)
        for subdir, _, files in os.walk(exposure_path):
            folder_name = os.path.basename(subdir)
            tiff_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff')) and "overlay" not in f.lower()])

            if not tiff_files:
                continue

            print(f"  Processing subfolder: {subdir}")
            for tiff_file in tiff_files:
                tiff_path = os.path.join(subdir, tiff_file)
                try:
                    img = Image.open(tiff_path)
                    img_array = np.array(img)
                    pixel_values = img_array.flatten()
                    saturated_pixels = np.sum(pixel_values == 65535)
                    histograms.append((tiff_file, subdir, img_array, pixel_values, saturated_pixels))
                    tiff_images.append((tiff_file, img_array))  # Add the image to the list
                except Exception as e:
                    print(f"Error processing {tiff_path}: {e}")

        if histograms:
            cols = 4
            rows = (len(histograms) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
            axes = np.array(axes).flatten()

            # Set number of Gaussian components
            num_components = 4  # we are using 4 GMM components

            for i, (tiff_file, subdir, img_array, pixel_values, saturated_pixels) in enumerate(histograms):
                ax1 = axes[i]
                ax2 = ax1.twinx()

                bins = np.linspace(0, 65535, 256) # creates an aray of evenly spaced values

                num_bins = 256  # Keep the number of bins the same
                bin_edges = np.linspace(0, 65535, num_bins + 1)  # Ensure 65535 has its bin
                hist_data, bin_edges = np.histogram(pixel_values, bins=bin_edges)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers to plot on the x-axis.

                # Fit Gaussian Mixture Model
                gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=0)
                gmm.fit(pixel_values.reshape(-1, 1))

                # Extract GMM parameters
                means = gmm.means_.flatten()
                variances = gmm.covariances_.flatten()
                responsibilities = gmm.predict_proba(pixel_values.reshape(-1, 1))
                assigned_components = np.argmax(responsibilities, axis=1).reshape(img_array.shape)

                # Sorting the components by mean intensity
                sorted_indices = np.argsort(means)  # Get indices of sorted means
                sorted_means = means[sorted_indices]  # Sorted means
                sorted_variances = variances[sorted_indices]  # Sorted variances
                sorted_colors = [custom_colors[i] for i in range(len(sorted_indices))]  # Ensuring consistent colors

                # Re-map assigned components to match the sorted order
                remapped_components = np.zeros_like(assigned_components)
                for new_index, old_index in enumerate(sorted_indices):
                    remapped_components[assigned_components == old_index] = new_index  # Re-mapping the indices

                # Assign colors to histogram bins based on sorted mapping above
                bin_colors = []
                bin_assignments = np.digitize(pixel_values, bins) - 1  # Find bin index for each pixel

                # Update this function to add 65535 condition for saturation
                for j in range(len(bin_centers)):
                    pixels_in_bin = pixel_values[bin_assignments == j]

                    # Check if any pixels == 65535
                    if bin_centers[j] >= 65535 or np.any(pixels_in_bin == 65535):  
                        bin_colors.append((255/255, 0/255, 255/255))  # Magenta color
                    else:
                        assigned_in_bin = remapped_components.flat[np.isin(pixel_values, pixels_in_bin)]
                        if len(assigned_in_bin) > 0:
                            component_counts = np.bincount(assigned_in_bin, minlength=num_components)
                            dominant_component = np.argmax(component_counts)
                            bin_colors.append(sorted_colors[dominant_component])
                        else:
                            bin_colors.append((0, 0, 0))  # Assigning default black color for (if empty bin)

                # Update histogram plot with sorted colors
                ax1.bar(bin_centers, hist_data, width=np.diff(bin_edges), color=bin_colors, alpha=0.7)
                ax2.bar(bin_centers, hist_data, width=np.diff(bin_edges), color=bin_colors, alpha=0.2, log=True)

                # Extract and format subfolder name
                subfolder_name = os.path.basename(subdir)
                # Define a mapping for the expected names
                subfolder_mapping = {
                        "XY01": "XY01-0.7",
                        "XY02": "XY02-1.1",
                        "XY03": "XY03-1.5",
                        "XY04": "XY04-1.9"
                    }

                # Update title with the formatted name
                formatted_name = subfolder_mapping.get(subfolder_name, subfolder_name)
                ax1.set_title(f"({formatted_name})", fontsize=5)

                # Label the X and Y axis
                ax1.set_ylim(0, 90000)
                ax1.set_ylabel("Pixel Count (Linear)", fontsize=5)
                ax2.set_ylabel("Pixel Count (Log Scale)", fontsize=5)
                ax1.set_xlabel("Pixel Intensity", fontsize=5)
                ax1.tick_params(axis='y', colors='black', labelsize=5) 
                ax1.tick_params(axis='x', colors='black', labelsize=5) 
                ax2.tick_params(axis='y', colors='gray', labelsize=5) 
                ax1.text(0.08, 0.85, f"Saturated pixels: {saturated_pixels}", transform=ax1.transAxes, fontsize=5, color="red")
                ax1.set_xticks([0, 65535])
                ax1.set_yticks([ax1.get_yticks()[0], ax1.get_yticks()[-1]])
                
                # Display mean and variance values
                for j, (mean, var, color) in enumerate(zip(sorted_means, sorted_variances, sorted_colors)):
                    ax1.text(0.5, 0.8 - j * 0.08, f"Mean: {mean:.2f}, Std: {np.sqrt(var):.2f}", 
                            transform=ax1.transAxes, fontsize=5, color=color)

                # Update Gaussian fit plots with sorted colors
                gmm_x = np.linspace(0, 65535, 256)
                for j, (mean, var) in enumerate(zip(sorted_means, sorted_variances)):
                    std_dev = np.sqrt(var)
                    gaussian_curve = norm.pdf(gmm_x, mean, std_dev) * len(pixel_values) * (bin_edges[1] - bin_edges[0])
                    ax1.plot(gmm_x, gaussian_curve, linestyle='dashed', alpha=0.7, color=sorted_colors[j], label=f"Gaussian {j+1}")

                # Create segmented image using remapped components
                segmented_image = create_segmented_image(remapped_components, sorted_colors)

                # Save segmented image
                segmented_path = os.path.join(exposure_path, f"{tiff_file}_segmented.png")
                plt.imsave(segmented_path, segmented_image)

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            fig.suptitle(f"{exposure_folder}", fontsize=7)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig) 
