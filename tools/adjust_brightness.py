import cv2
import numpy as np
import os
import sys

def adjust_brightness(image, target_brightness=150):
    """
    Adjusts the brightness of an image to a target brightness level.

    :param image: Input image as a NumPy array.
    :param target_brightness: Desired average brightness level (0-255).
    :return: Brightened image as a NumPy array.
    """

    if image is None:
        raise ValueError("Invalid image: The input image could not be read.")

    # Convert to grayscale to calculate current brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)  # Ensure safe numerical operations
    current_brightness = np.mean(gray)

    if current_brightness == 0:
        current_brightness = 1e-5  # Small positive value to avoid division by zero
    # Calculate the brightness factor
    brightness_factor = target_brightness / current_brightness

    # Adjust the brightness
    brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    return brightened_image

def process_images_from_folders(input_root_folders, output_root_folder, target_brightness=150):
    """
    Adjusts the brightness of all images in multiple folders and saves the results.

    Arguments
    ---------
        input_folders: 
            List of paths to folders containing input images.

        output_root_folder: 
            Path to the root folder to save brightened images.

        target_brightness: 
            Desired average brightness level (0-255).
    """
    for root, dirs, files in os.walk(input_root_folders):
        # Create the corresponding output folder structure
        relative_path = os.path.relpath(root, input_root_folders)
        output_folder = os.path.join(output_root_folder, relative_path)
        # folder_name = os.path.basename(folder.rstrip('/\\'))  # Get folder name
        # output_folder = os.path.join(output_root_folder, folder_name)
        print(relative_path, output_folder)

        os.makedirs(output_folder, exist_ok=True)

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Read the image
                img_path = os.path.join(root, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue

                # Adjust brightness
                try:
                    brightened_image = adjust_brightness(image, target_brightness)

                    # Save the image
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, brightened_image)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    process_images_from_folders(sys.argv[1] , sys.argv[2], float(sys.argv[3]))
    print("Brightness adjustment complete.")