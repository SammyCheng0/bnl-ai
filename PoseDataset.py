import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from torchvision.transforms import functional as F
from torchvision.transforms import Pad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import cv2
import numpy as np

def calculate_padding(original_width, original_height, target_width, target_height):
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if original_aspect_ratio > target_aspect_ratio:
        # Pad height
        new_height = original_width / target_aspect_ratio
        padding_height = (new_height - original_height) / 2
        padding_width = 0
    else:
        # Pad width
        new_width = original_height * target_aspect_ratio
        padding_width = (new_width - original_width) / 2
        padding_height = 0

    return int(padding_width), int(padding_height)

# def calculate_dataset_mean_std(dataset, batch_size=1):
#     """
#     Calculates the mean and standard deviation of an image dataset.
    
#     Args:
#         dataset (Dataset): PyTorch dataset object.
#         batch_size (int): Batch size for processing images.

#     Returns:
#         mean (torch.Tensor): Mean per channel (R, G, B).
#         std (torch.Tensor): Standard deviation per channel (R, G, B).
#     """
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
#     mean_sum = torch.zeros(3)
#     std_sum = torch.zeros(3)
#     num_pixels = 0

#     for images, _, _ in data_loader:
#         batch_pixels = images.shape[1] * images.shape[2]  # Batch * H * W
#         mean_sum += images.mean(dim=[0, 1, 2]) * batch_pixels
#         std_sum += images.std(dim=[0, 1, 2]) * batch_pixels
#         num_pixels += batch_pixels

#     mean = mean_sum / num_pixels
#     std = std_sum / num_pixels
#     return mean, std

def calculate_dataset_mean_std(dataset, batch_size=1):
    """
    Calculates the mean and standard deviation of an image dataset.

    Args:
        dataset (Dataset): PyTorch dataset object.
        batch_size (int): Batch size for processing images.

    Returns:
        mean (torch.Tensor): Mean per channel (R, G, B).
        std (torch.Tensor): Standard deviation per channel (R, G, B).
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    num_pixels = 0

    for images, _, _ in data_loader:
        # Ensure images are in the shape [batch_size, channels, height, width]
        print(images.size(1))
        batch_pixels = images.size(2) * images.size(3)  # H * W per image (image height and width)

        # Mean and standard deviation calculation across all pixels and channels
        mean_sum += images.mean(dim=[1, 2, 3]) * images.size(1)  # Mean per channel (R, G, B)
        std_sum += images.std(dim=[1, 2, 3]) * images.size(1)  # Std per channel (R, G, B)
        num_pixels += batch_pixels * images.size(1)  # Total number of pixels in the batch
        print("mean_sum: ", mean_sum)
        print("std_sum: ",std_sum)
        print("num_pixels: ", num_pixels)

    # Final mean and std across all images
    mean = mean_sum / num_pixels
    print("mean: ",mean)
    std = std_sum / num_pixels
    print("std: ", std)
    return mean, std


class PoseDataset(Dataset):
    def __init__(self, image_folder, resize_to, heatmap_size, label_file=None, rotate=False, mean=None, std=None):
    # def __init__(self, image_folder, resize_to, heatmap_size, label_file=None, rotate=False):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            label_file (str): Path to the CSV file containing labels (keypoints).
            resize_to (tuple): Imaga size to which the transformation will resize to (height, width).
            rotate (bool): Whether to randomly rotate the images of 90, 180 or 270 degrees.
        """
        self.image_folder = image_folder
        self.resize_to = resize_to
        self.heatmap_size = heatmap_size
        self.rotate = rotate
        self.labels = pd.read_csv(label_file)
        self.filenames = os.listdir(image_folder)
        self.filenames.sort()
        self.mean = mean
        self.std = std

        # print(self)

        # img_name = self.filenames[idx]

        # idx = self.labels[self.labels['filename'] == img_name].index[0]

        # bbox = self.bbox.iloc[idx,:].to_numpy()
        
        # img_path = os.path.join(self.image_folder, img_name)
        # transformed_image = Image.open(img_path).convert("RGB")

        # bbox
        df = self.labels.copy()
        self.bbox = df[[col for col in df.columns if col.startswith("bbox_")]]
        
        # keypoints
        df = df.drop(columns=[col for col in df.columns if col.startswith("bbox_")])
        keypoints = sorted(set(col.rsplit('-', 1)[0] for col in df.columns if '-' in col))
        ordered_columns = [col for pair in keypoints for col in (f"{pair}-x", f"{pair}-y") if col in df.columns]
        ordered_columns = ['filename'] + ordered_columns if 'filename' in df.columns else ordered_columns
        self.keypoints = df[ordered_columns]

        # bodypart-heatmap index
        df_cp = df.copy()
        df_cp = df_cp.drop(columns=['filename'])
        self.bp_hm_index = {}
        for idx, col in enumerate(df_cp.columns):
            # Remove '-x' or '-y' from the column name
            base_name = col.rstrip('-xy')

            if base_name not in self.bp_hm_index:
                self.bp_hm_index[base_name] = idx

        if mean is None or std is None:
            self.mean, self.std = calculate_dataset_mean_std(self)
        else:
            self.mean = mean
            self.std = std
        # print('bodypart-heatmap index')
        # print(self.bp_hm_index) 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        # ------- steps --------
        # Crop image around mouse
        # rotate with extension = True
        # add padding so that matches next resizing aspect ratio
        # resize to match input model
        # normalize image
        # generate heatmap for each keypoint
        # ----------------------

        img_name = self.filenames[idx]

        idx = self.labels[self.labels['filename'] == img_name].index[0]

        bbox = self.bbox.iloc[idx,:].to_numpy()
        
        img_path = os.path.join(self.image_folder, img_name)
        transformed_image = Image.open(img_path).convert("RGB")
            # original_image = np.array(transformed_image.copy())
        keypoints = self.keypoints[self.keypoints.iloc[:, 0] == img_name].iloc[:, 1:].values.astype('float32') # Rest are keypoints
        # keypoints = self.keypoints.loc[img_name, 1:].values.astype('float32')  # Rest are keypoints
        keypoints = torch.tensor(keypoints)
        keypoints = keypoints.view(-1, 2)

        # ----------------------------------
        # --- Crop images and keypoints ----
        # ----------------------------------

        # if not self.infer:
        # Filter out invalid (NaN) keypoints
        valid_keypoints = keypoints[~torch.isnan(keypoints).any(dim=1)]
        if len(valid_keypoints) == 0:
            raise ValueError("All keypoints are NaN for this sample.")
        transformed_image = transformed_image.crop(bbox)

        # Crop keypoints
        keypoints[:, 0] -= bbox[0]
        keypoints[:, 1] -= bbox[1]
        keypoints = keypoints.view(-1)

        # ----------------------------------
        # --- Rotate images and keypoints --
        # ----------------------------------

        keypoints = keypoints.view(-1, 2)
        # Random rotation
        if self.rotate:
            angle = random.choice([0, 90, 180, 270])
            # Calculate bounding box of the rotated image and rotate image
            crop_width, crop_height = transformed_image.size
            angle_rad = math.radians(angle)
            transformed_image = F.rotate(transformed_image, angle, expand=True)
            # Rotate keypoints
            center_x, center_y = transformed_image.size[0] / 2, transformed_image.size[1] / 2
            rotation_matrix = torch.tensor([
                [math.cos(-angle_rad), -math.sin(-angle_rad)],
                [math.sin(-angle_rad), math.cos(-angle_rad)]
            ])
            keypoints += torch.tensor([(transformed_image.size[0] - crop_width) / 2, (transformed_image.size[1] - crop_height) / 2])  # Adjust for padding
            keypoints -= torch.tensor([center_x, center_y])
            keypoints = torch.mm(keypoints, rotation_matrix.T) + torch.tensor([center_x, center_y])
        

        # ----------------------------------
        # --- Add padding and resize -------
        # ----------------------------------

        # Percentage of scaling of the cropped mouse photo
        scale_size = round(random.uniform(0.9 , 1.2), 1)

        image_width, image_length = transformed_image.size
        transformed_width = round(image_width*scale_size, 0)
        transformed_length = round(image_length*scale_size, 0)
        # resizing the mouse photo with the scale
        transformed_image = transformed_image.resize((int(transformed_width), int(transformed_length)))
        
        # Resize keypoints for the scaling
        keypoints[::2] *= scale_size  # Scale x-coordinates
        keypoints[1::2] *= scale_size  # Scale y-coordinates

        # Calculate padding to match the aspect ratio
        padding_width, padding_height = calculate_padding(*transformed_image.size, self.resize_to[1], self.resize_to[0])
        # print(f"padding width: {padding_width}, padding height: {padding_height}")
        transformed_image = Pad(padding=(padding_width, padding_height), fill=0, padding_mode='constant')(transformed_image)
        # Add padding to keypoints
        keypoints += torch.tensor([padding_width, padding_height])  # Adjust for padding
        keypoints = keypoints.view(-1)
        # Resize image to output size
        scale_x = self.resize_to[1] / transformed_image.size[0]
        scale_y = self.resize_to[0] / transformed_image.size[1]
        transformed_image = F.resize(transformed_image, self.resize_to)
        # Resize keypoints
        keypoints[::2] *= scale_x  # Scale x-coordinates
        keypoints[1::2] *= scale_y  # Scale y-coordinates
        padding_width_hm  = int( padding_width * scale_x)
        padding_height_hm = int( padding_height * scale_y)

        transformed_image = F.to_tensor(transformed_image)

        def get_motion_blur_kernel(x, y, thickness=1, ksize=11):
            """ Obtains Motion Blur Kernel
                Inputs:
                    x - horizontal direction of blur
                    y - vertical direction of blur
                    thickness - thickness of blur kernel line
                    ksize - size of blur kernel
                Outputs:
                    blur_kernel
                """
            blur_kernel = np.zeros((ksize, ksize))
            c = int(ksize/2)

            blur_kernel[c, :] = 255  

            blur_kernel /= ksize 
            return blur_kernel

        # Convert image to NumPy array
        transformed_image = np.array(transformed_image)

        # Make blur kernel and apply it
        image_width, image_length = transformed_image.shape[1], transformed_image.shape[0]
        blur_kernel = get_motion_blur_kernel(x=0, y=0, thickness=1, ksize=3)
        # Apply motion blur using filter2D
        transformed_image = cv2.filter2D(transformed_image, ddepth=-1, kernel=blur_kernel)

        # If you need to convert back to a tensor, do it after processing
        transformed_image = torch.from_numpy(transformed_image).float() / 255.0
        # if self.mean is not None and self.std is not None:
        #     transformed_image = F.normalize(transformed_image, mean=self.mean, std=self.std)

        # normalize image
        # mean_image = transformed_image.mean(dim=[1, 2])  # Mean per channel (R, G, B)
        # std_image = transformed_image.std(dim=[1, 2])  # Std per channel (R, G, B)

        # transformed_image = F.normalize(transformed_image, mean= mean_image, std= std_image)
        # mean_image = ([0.1058, 0.1058, 0.1058])
        # std_image = ([0.1372, 0.1372, 0.1372])
        # transformed_image = F.normalize(transformed_image, mean= mean_image, std= std_image)



        # ----------------------------------
        # ------- Generate heatmaps --------
        # ----------------------------------
        heatmaps = []
        keypoints = keypoints.view(-1, 2)
        for keypoint in keypoints:
            heatmap = generate_heatmap(transformed_image, keypoint, 
                                       padding_width=padding_width_hm,
                                       padding_height=padding_height_hm,
                                       heatmap_size=self.heatmap_size, sigma=0.8)
            heatmaps.append(heatmap)

        heatmaps = torch.stack(heatmaps)

        return transformed_image, keypoints, heatmaps

def generate_heatmap(image, keypoint, padding_width, padding_height, heatmap_size=(64, 48), sigma=1):
    """
    Generates a heatmap for a given keypoint while handling black padding.

    Args:
        image (torch.Tensor): Original image tensor of shape (C, H, W).
        keypoint (torch.Tensor): Keypoint tensor of shape (2,), in (x, y) format.
        padding_width (int): Width of the padding.
        padding_height (int): Height of the padding.
        heatmap_size (tuple): Output heatmap size (height, width).
        sigma (float): Standard deviation for the Gaussian.

    Returns:
        torch.Tensor: Heatmap tensor of shape (1, height, width).
    """
    # Unpack dimensions
    _, img_h, img_w = image.shape
    heatmap_h, heatmap_w = heatmap_size

    # Check for NaN values in keypoint
    if torch.isnan(keypoint).any():
        return torch.zeros(heatmap_h, heatmap_w, dtype=torch.float32)

    # Convert keypoint to heatmap space
    x, y = keypoint
    scale_x = heatmap_w / img_w
    scale_y = heatmap_h / img_h
    keypoint_hm = torch.tensor([x * scale_x, y * scale_y])
    padding_width  = int(padding_width * scale_x)
    padding_height = int(padding_height * scale_y)

    # Create the Gaussian heatmap
    heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
    center_x, center_y = int(keypoint_hm[0]), int(keypoint_hm[1])

    for i in range(heatmap_h):
        for j in range(heatmap_w):
            heatmap[i, j] = np.exp(-((i - center_y) ** 2 + (j - center_x) ** 2) / (2 * sigma ** 2))

    if padding_height != 0:
        heatmap[:padding_height, :]  = 0  # Top padding
        heatmap[-padding_height:, :] = 0  # Bottom padding
    if padding_width != 0:
        heatmap[:, :padding_width]   = 0  # Left padding
        heatmap[:, -padding_width:]  = 0  # Right padding

    # heatmap += 1e-10

    # Normalize heatmap to range [0, 1]
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    # Convert to tensor
    return torch.tensor(heatmap, dtype=torch.float32)

if __name__ == "__main__":
    # Set paths
    image_folder = r"images"
    label_file = r"sorted_annotations.csv"
    # label_file = r"annotations.csv"

    # Create dataset and data loader
    dataset = PoseDataset(image_folder=image_folder,
                          heatmap_size=(64, 48),
                          label_file=label_file, 
                          resize_to=(256, 192))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # transformed_image = Image.open(img_path).convert("RGB")
    # images = 
    # return transformed_image, keypoints, heatmaps

    # mean_sum = torch.zeros(3)
    # std_sum = torch.zeros(3)
    # num_pixels = 0

    # for images, _, _ in dataset:
    #     batch_pixels = images.shape[1] * images.shape[2]  # H * W
    #     mean_sum += images.mean(dim=[0, 1, 2]) * batch_pixels
    #     std_sum += images.std(dim=[0, 1, 2]) * batch_pixels
    #     num_pixels += batch_pixels

    # mean_images = mean_sum / num_pixels
    # std_images = std_sum / num_pixels

    # transformed_images = []

    # for images, _, _ in dataset:
    #     transformed_image = F.normalize(images, mean= mean_images, std= std_images)
    #     transformed_images.append(transformed_image)

# return transformed_image, keypoints, heatmaps
    
    # print(data_loader)

    # Iterate through the data loader
    # selec a random integer between 0 and dataset length
    index = random.randint(0, len(dataset))
    for i, (images, gt_kps, gt_hms) in enumerate(data_loader):
        if i % 5 == 0:  # Show every 10th batch
            image = images[0]
            gt_hms = gt_hms[0]

            overlap_hm = gt_hms[0]
            for hm in gt_hms[1:]:
                try:
                    overlap_hm = torch.maximum(overlap_hm, hm)        
                except Exception as e: 
                    if overlap_hm is None:
                        overlap_hm = hm

            fig, ax = plt.subplots(1, 2)
            ax[1].imshow(overlap_hm, cmap='hot', interpolation='nearest')
            ax[0].imshow(image.numpy().transpose(1, 2, 0))
            # ax[1].scatter(keypoints[:,0], keypoints[:,1], c='red', s=20)  # Plot keypoints
            # ax[0].imshow(cropped_image[0])
            plt.show()

