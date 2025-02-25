import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

# Define main root directory where all exp-time subfolders are present
main_root_dir = r"C:\Users\syagi\Documents\CH2"

# Define a single PDF filename to store all histogram plots
pdf_filepath = os.path.join(main_root_dir, "LinearandLog.pdf")

# Open PDF for saving
with PdfPages(pdf_filepath) as pdf:
    print(f"Saving all histograms in {pdf_filepath}")

    # Loop through each exp-time folder
    for exposure_folder in sorted(os.listdir(main_root_dir)):  # Sort for consistency
        exposure_path = os.path.join(main_root_dir, exposure_folder)

        # Condition to check if it's a directory (skip files)
        if not os.path.isdir(exposure_path):
            continue  

        print(f"Processing exposure folder: {exposure_folder}")

        histograms = []  # Store histogram plots for this folder

        # Loop through subfolders (XY01, XY02, etc.)
        for subdir, _, files in os.walk(exposure_path):
            tiff_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])

            if not tiff_files:
                continue  # Skip folders without TIFF files
            
            print(f"  Processing subfolder: {subdir}")

            for tiff_file in tiff_files:
                tiff_path = os.path.join(subdir, tiff_file)

                try:
                    # Open the image
                    img = Image.open(tiff_path)

                    # Convert to NumPy array
                    img_array = np.array(img)

                    # Flatten to get pixel intensity values
                    pixel_values = img_array.flatten()

                    # count saturated pixels
                    saturated_pixels = np.sum(pixel_values == 65535)

                    # Storing the data for later plotting
                    histograms.append((tiff_file, subdir, pixel_values, saturated_pixels))

                except Exception as e:
                    print(f"Error processing {tiff_path}: {e}")

        # **Only create a figure if there are histograms**
        if histograms:
            cols = 4  # Number of columns per row
            rows = (len(histograms) + cols - 1) // cols  # Compute required rows

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))  # Adjust figure size
            axes = np.array(axes).flatten()  # Flatten axes for easy indexing

            for i, (tiff_file, subdir, pixel_values, saturated_pixels) in enumerate(histograms):
                
                ax1 = axes[i] # setting the primary axis for linear plot
                ax2 = ax1.twinx() # axis for log plot

                bins = np.linspace(0, 65535, 256)

                # linear
                ax1.hist(pixel_values, bins=bins, color='black', alpha=1, label="Linear Scale")
                ax1.set_ylabel("Pixel Count (Linear)", color='black', fontsize=7) # can be removed if not needed
                ax1.tick_params(axis='y', colors='black', labelsize=7)
                

                # log
                ax2.hist(pixel_values, bins=bins, color='gray', alpha=0.6, log=True, label="Log Scale")
                ax2.set_ylabel("Pixel Count (Log Scale)", color='gray', fontsize=7)
                ax2.tick_params(axis='y', colors='gray', labelsize=7)
               

                # title
                ax1.set_title(f"({os.path.basename(subdir)})", fontsize=7)
                # x-axis label
                ax1.set_xlabel("Pixel Intensity", fontsize=7)
                ax1.tick_params(axis='x', colors='black', labelsize=7)

                ax1.set_xticks([0, 65535])
                ax1.set_yticks([ax1.get_yticks()[0], ax1.get_yticks()[-1]])

                # to display saturated pixel count
                ax1.text(0.3,0.85, f"Saturated pixels: {saturated_pixels}",  # x and y position to place the figure
                            transform=ax1.transAxes, fontsize=7, color="red")
                           # bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            fig.suptitle(f"Histograms for {exposure_folder}", fontsize=8)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout

            # Save the figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)
pdf.close()
print(f"All histograms saved in {pdf_filepath}")

