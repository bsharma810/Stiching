import os
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

# List of TIFF files
tif_files = [f"Image_XY01_{str(i).zfill(5)}_CH4.tif" for i in range(1, 10)]  # Adjust range based on your files

# Function to extract XML from TIFF and parse coordinates (X and Y) and dimensions (width & height)
def extract_coordinates_and_dimensions(tif_files):
    image_positions = {}
    dimensions = {}
    
    for tif in tif_files:
        xml_file = tif.replace('.tif', '.xml')
        # Extract XML content from TIFF file
        subprocess.run(["grep", "-a", "Data", "-B0", "-A1000", tif], stdout=open(xml_file, 'w'))

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

            # Print the extracted information
            print(f"File: {xml_file} | X: {x}, Y: {y}, Width: {width}, Height: {height}")
        
        else:
            print(f"File: {xml_file} | Attributes not found")
    
    return image_positions, dimensions

# Stitching function
def stitch_images(image_positions, new_order, nm_per_pixel, output_filename):
    # Convert coordinates from nanometers to pixels
    image_positions_px = {
        filename: (int(x_nm / nm_per_pixel), int(y_nm / nm_per_pixel))
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

# Run the extraction and stitching process
image_positions, dimensions = extract_coordinates_and_dimensions(tif_files)

# Conversion factor: nm to pixels
nm_per_pixel = 1887.21

# Output stitched image filename
output_filename = 'Automated-stitched-image.tif'

# Stitch images using the merged array
stitch_images(image_positions, list(image_positions.keys()), nm_per_pixel, output_filename)



