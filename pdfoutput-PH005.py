import subprocess
import glob
import xml.etree.ElementTree as ET
import re
from PIL import Image
import numpy as np
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# Folder path
main_folder_path = r"C:\Users\syagi\Documents\Project-proteomics\PH005"

# function to save images to a pdf file with labels
def save_to_pdf(image_files, output_pdf, dpi=300):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    page_width, page_height = A4  # 612 x 792 points (8.5 x 11 inches)

    cols, rows = 1, 2  # Grid layout (2x2)
    margin = 50
    spacing_x, spacing_y = 20, 20  # Space between images

    # Max image size based on page size
    max_img_width = (page_width - 2 * margin - (cols - 1) * spacing_x) / cols
    max_img_height = (page_height - 2 * margin - (rows - 1) * spacing_y) / rows

    x_start = margin
    y_start = page_height - margin  # Start from the top-left

    count = 0
    for img_path, label in image_files:
        if count % (cols * rows) == 0:
            if count > 0:
                c.showPage()  # Start a new page
            c.setFont("Helvetica-Bold", 14)
            c.drawCentredString(page_width / 2, page_height - 40, f"Folder: {folder}")

        img = Image.open(img_path)
        img = img.convert("RGB")  # Ensure correct format

        # Calculate scaling factor to maintain aspect ratio
        img_width, img_height = img.size
        scale_factor = min(max_img_width / img_width, max_img_height / img_height)

        new_width = img_width * scale_factor
        new_height = img_height * scale_factor

        img_reader = ImageReader(img)

        col_index = count % cols
        row_index = (count // cols) % rows  # Reset row after `cols` images
              
        x_pos = x_start + col_index * (max_img_width + spacing_x)
        y_pos = y_start - (row_index + 1) * (max_img_height + spacing_y)

        # Draw label
        # Draw label
        c.setFont("Helvetica", 10)
        c.drawString(x_pos, y_pos + new_height + 5, label)

        # Draw image with adjusted size
        c.drawImage(img_reader, x_pos, y_pos, width=new_width, height=new_height, preserveAspectRatio=True, mask='auto')

        count += 1
    c.save()
    print(f"saved stiched images to {output_pdf}")

# Function to extract XML from TIFF and parse coordinates (X and Y) and dimensions (width & height)
def extract_coordinates_and_dimensions(tif_files):
    image_positions = {}
    dimensions = {}
    nm_per_pixel_values = {}

    for tif in tif_files:
        xml_file = tif.replace('.tif', '.xml')
      
        # Read TIFF file as binary
        with open (tif, "rb") as file:
            content = file.read().decode(errors="ignore")   # decode as string
                  
        # Extract XML content from TIFF file (file starts with <Data> and ends with <\Data>)
        match = re.search(r"<Data>.*?</Data>", content, re.DOTALL)

        if match: 
            xml_content = match.group(0)  # extract matched xml content
            with open(xml_file, "w", encoding="utf-8") as xml_out:
                xml_out.write(xml_content) 
        else:
            print(f"Could not extract XML data from {tif}")
            continue
           
        # Parse the XML file and extract coordinates and dimensions
        tree = ET.parse(xml_file)
        region = tree.find('.//XyStageRegion')
        
        if region is not None:
            # Extract X and Y coordinates
            x = int(region.find('X').text)
            y = int(region.find('Y').text)
            image_positions[tif] = (x, y)

            # Extract width and height from the SavingImageSize section
            width = int(tree.find('.//SavingImageSize/Width').text)
            height = int(tree.find('.//SavingImageSize/Height').text)
            dimensions[tif] = (width, height)

            # Extract width from XyStageRegion and SavingImageSize to calculate nm_per_pixel
            xy_stage_width = int(region.find('Width').text)  # Width in nm from XyStageRegion
            saving_image_width = width  # Width in pixels from SavingImageSize
            
            nm_per_pixel = xy_stage_width / saving_image_width  # Conversion factor
            nm_per_pixel_values[tif] = nm_per_pixel

            # Print the extracted information
            print(f"File: {xml_file} | X: {x}, Y: {y}, Width: {width}, Height: {height}, nm_per_pixel: {nm_per_pixel}")
        
        else:
            print(f"File: {xml_file} | Attributes not found")
    
    return image_positions, dimensions, nm_per_pixel_values

# Stitching function
def stitch_images(image_positions, new_order, nm_per_pixel_values, output_filename):
    # Convert coordinates from nanometers to pixels
    image_positions_px = {
        filename: (int(x_nm / nm_per_pixel_values[filename]), int(y_nm / nm_per_pixel_values[filename]))
        for filename, (x_nm, y_nm) in image_positions.items()
    }

    # Calculate canvas size based on min/max coordinates
    min_x = min(pos[0] for pos in image_positions_px.values())
    min_y = min(pos[1] for pos in image_positions_px.values())
    max_x = max(pos[0] for pos in image_positions_px.values()) + dimensions[new_order[0]][0]  # Add width of one image
    max_y = max(pos[1] for pos in image_positions_px.values()) + dimensions[new_order[0]][1]  # Add height of one image

    # Canvas dimensions (blank array for merged image data)
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    canvas_array = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # Merged array for the canvas

    # Place images in the defined new order
    for idx, filename in enumerate(new_order, start=1):
        if filename not in image_positions_px:
            print(f"File not found: {filename}")
            continue

        # Open the image and handle different modes
        img = Image.open(filename)
        
        # Check the mode of the image and handle accordingly
        if img.mode == 'RGB':
            img_array = np.array(img)
        elif img.mode == 'L':
            img = img.convert('RGB')
            img_array = np.array(img)
        elif img.mode == 'I;16':
            img = img.point(lambda i: i * (1/256)).convert('RGB')  # Normalize to 0-255
            img_array = np.array(img)
        elif img.mode == 'RGBA':
            img = img.convert('RGB')
            img_array = np.array(img)
        else:
            print(f"Unsupported image mode: {img.mode}. Skipping {filename}.")
            continue  # Skip to the next image if unsupported

        # Get the original position for this image
        x_pos_px, y_pos_px = image_positions_px[filename]
        
        # Retrieve width and height from dimensions dictionary
        width, height = dimensions[filename]
        
        # Calculate the position relative to the bottom-right corner of the canvas
        x_relative = max_x - (x_pos_px + width)
        y_relative = max_y - (y_pos_px + height)

        # Print the paste location
        print(f"Pasting {filename} at ({x_relative}, {y_relative})")

        # Place the image onto the canvas array
        canvas_array[y_relative:y_relative + height, x_relative:x_relative + width] = img_array

    # Convert the canvas array back into an image
    stitched_image = Image.fromarray(canvas_array)

    # Save and show the stitched image
    stitched_image.save(output_filename)
    stitched_image.show()
    return stitched_image

# Function to overlay stitched images from different channels
def overlay_images(stitched_images, output_filename):
    # Convert the first image to RGBA
    base_image = stitched_images[0].convert("RGBA")
    
    # Resize all images to the base image's dimensions
    base_width, base_height = base_image.size
    resized_images = [base_image]

    for img in stitched_images[1:]:
        img = img.convert("RGBA")
        # Resize the current image to match the base image's dimensions
        img = img.resize((base_width, base_height), resample=Image.Resampling.LANCZOS)
        resized_images.append(img)

    # Perform the overlay with resized images
    for img in resized_images[1:]:
        base_image = Image.alpha_composite(base_image, img)

    base_image.save(output_filename)
    base_image.show()
    return output_filename


# Loop through folders under the main folder path
all_subfolders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
pdf_image_list = []

# Process each folder
for folder in all_subfolders:
    folder_path = os.path.join(main_folder_path, folder)
    print(f"Processing folder: {folder_path}")
    
    all_tif_files = sorted(glob.glob(os.path.join(folder_path, "Image_XY01_*.tif")))
    channels = sorted(
        set([os.path.basename(filename).split("_")[-1].replace(".tif", "") for filename in all_tif_files]),
        key=lambda x: int(re.search(r'CH(\d+)', x).group(1)) if re.search(r'CH(\d+)', x) else float('inf')
    )
    
    stitched_images = []
    pdf_image_list_folder = []
    
    for channel in channels:
        print(f"Processing channel: {channel} in folder {folder}")
        tif_files = sorted([f for f in all_tif_files if f.endswith(f"{channel}.tif")])
        image_positions, dimensions, nm_per_pixel_values = extract_coordinates_and_dimensions(tif_files)
        
        output_filename = os.path.join(folder_path, f"Stitched_{channel}.tif")
        stiched_img = stitch_images(image_positions, tif_files, nm_per_pixel_values, output_filename)
        stitched_images.append(stiched_img)
        
        pdf_image_list_folder.append((output_filename, f"Stitched Image - Channel {channel}"))

    # Overlay stitched images from all channels
    overlay_filename = os.path.join(folder_path, f"Overlay_{folder}.tif")
    overlay_images(stitched_images, overlay_filename)
    #pdf_image_list_folder.append((overlay_filename, f"Overlay Image - {folder}"))

    # Save folder's results to a PDF page
    save_to_pdf(pdf_image_list_folder, os.path.join(folder_path, f"{folder}_Results.pdf"))
    pdf_image_list.append((os.path.join(folder_path, f"{folder}_Results.pdf"), f"Folder: {folder}"))

# find all TIFF files in specified folder
all_tif_files = sorted(glob.glob(os.path.join(folder_path, "Image_XY01_*.tif")))

# Identify unique channels
#channels = set([os.path.basename(filename).split("_")[-1].replace(".tif", "") for filename in all_tif_files])
channels = sorted(
    set([os.path.basename(filename).split("_")[-1].replace(".tif", "") for filename in all_tif_files]),
    key=lambda x: int(re.search(r'CH(\d+)', x).group(1)) if re.search(r'CH(\d+)', x) else float('inf')
)

stitched_images = []
pdf_image_list = []

for channel in channels:
    print(f"Processing channel: {channel} ")
    # assign tiff files
    tif_files = sorted([f for f in all_tif_files if f.endswith(f"{channel}.tif")])
    # get all images for this channel
    image_positions, dimensions, nm_per_pixel_values = extract_coordinates_and_dimensions(tif_files)
    # output filename for stiched channel
    output_filename = f"Stitched_{channel}.tif"
    # stich images and store result
    stiched_img = stitch_images(image_positions, tif_files, nm_per_pixel_values, output_filename)
    stitched_images.append(stiched_img)

    # Add to PDF image list
    pdf_image_list.append((output_filename, f"Stitched Image - Channel {channel}"))


# Final PDF to combine results from all folders
final_pdf_output = os.path.join(folder_path, "Final_Results.pdf")
save_to_pdf(pdf_image_list, final_pdf_output)
