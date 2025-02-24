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
from PIL import ImageChops

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

class TopViewDataset(Dataset):
    def __init__(self, image_folder, output_size, label_file=None, rotate=True, infer=False, debug=False):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            label_file (str): Path to the CSV file containing labels (keypoints).
            output_size (tuple): Output size of the images (height, width).
            rotate (bool): Whether to randomly rotate the images of 90, 180 or 270 degrees.
            infer (bool): Whether to use the dataset for inference.
            debug (bool): Whether to return information for debug.
        """
        self.image_folder = image_folder
        self.output_size = output_size
        self.debug = debug
        self.infer = infer
        self.rotate = rotate
        if infer:
            self.labels = None
        else:
            self.labels = pd.read_csv(label_file)
            self.labels.drop(labels=['tail_lower_midpoint-x', 'tail_lower_midpoint-y', 'tail_upper_midpoint-x', 'tail_upper_midpoint-y'], axis=1, inplace=True)
        self.filenames = os.listdir(image_folder)
        self.filenames.sort()

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

        # these are list because the dataloader does not support null values
        keypoints = []
        original_image = [] 
        not_normalized_image = []
         
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_folder, img_name)
        transformed_image = Image.open(img_path).convert("RGB")
        if self.debug:
            original_image = np.array(transformed_image.copy())
        if not self.infer:
            keypoints = self.labels[self.labels.iloc[:, 0] == img_name].iloc[:, 1:].values.astype('float32') # Rest are keypoints
            # keypoints = self.labels.loc[img_name, 1:].values.astype('float32')  # Rest are keypoints
            keypoints = torch.tensor(keypoints)
            keypoints = keypoints.view(-1, 2)

        original_keypoints = self.labels[self.labels.iloc[:, 0] == img_name].iloc[:, 1:].values.astype('float32')

        # ----------------------------------
        # --- Crop images and keypoints ----
        # ----------------------------------

        if not self.infer:
            # Filter out invalid (NaN) keypoints
            valid_keypoints = keypoints[~torch.isnan(keypoints).any(dim=1)]
            if len(valid_keypoints) == 0:
                raise ValueError("All keypoints are NaN for this sample.")
            min_x, _ = torch.min(valid_keypoints[:, 0], 0)
            max_x, _ = torch.max(valid_keypoints[:, 0], 0)
            min_y, _ = torch.min(valid_keypoints[:, 1], 0)
            max_y, _ = torch.max(valid_keypoints[:, 1], 0)
            # Add padding as 5% of the bounding box dimensions
            padding_x = 0.20 * (max_x - min_x)
            padding_y = 0.20 * (max_y - min_y)
            min_x = max(0, int(min_x - padding_x))
            min_y = max(0, int(min_y - padding_y))
            max_x = min(transformed_image.size[0], int(max_x + padding_x))
            max_y = min(transformed_image.size[1], int(max_y + padding_y))
            transformed_image = transformed_image.crop((min_x, min_y, max_x, max_y))

            # Crop keypoints
            keypoints[:, 0] -= min_x
            keypoints[:, 1] -= min_y
            keypoints = keypoints.view(-1)

        # ----------------------------------
        # --- Rotate images and keypoints --
        # ----------------------------------

        keypoints = keypoints.view(-1, 2)
        # Random rotation
        if self.rotate:
            # angle = random.uniform(-self.rotation, self.rotation)
            angle = random.choice([90, 180, 270])
            # Calculate bounding box of the rotated image and rotate image
            crop_width, crop_height = transformed_image.size
            angle_rad = math.radians(angle)
            transformed_image = F.rotate(transformed_image, angle, expand=True)
            # plt.plot(transformed_image)
            # plt.show()
            if not self.infer:
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

        scale_size = round(random.uniform(0.9 , 1.2), 1)
        print("scale: ", scale_size)

        image_width, image_length = transformed_image.size
        print("old: ", transformed_image.size)
        transformed_width = round(image_width*scale_size, 0)
        transformed_length = round(image_length*scale_size, 0)
        print("new: ", transformed_width, transformed_length)
        
        # Calculate padding to match the aspect ratio
        padding_width, padding_height = calculate_padding(*transformed_image.size, *self.output_size)
        print("padding old", padding_width, padding_height)

        transformed_image = transformed_image.resize((int(transformed_width), int(transformed_length)))
        print("resized: ",transformed_image.size)

        padding_width_old, padding_height_old = calculate_padding(*transformed_image.size, *self.output_size)
        print("padding new: ", padding_width_old, padding_height_old)

        transformed_image = Pad(padding=(padding_width, padding_height), fill=0, padding_mode='constant')(transformed_image)
        if not self.infer:
            # Add padding to keypoints
            keypoints += torch.tensor([padding_width, padding_height])  # Adjust for padding
            keypoints = keypoints.view(-1)

        
        scale_x = self.output_size[1] / transformed_image.size[0]
        scale_y = self.output_size[0] / transformed_image.size[1]
        transformed_image = F.resize(transformed_image, self.output_size)
        if not self.infer:
            # Resize keypoints
            keypoints[::2] *= scale_x  # Scale x-coordinates
            keypoints[1::2] *= scale_y  # Scale y-coordinates
            padding_width_hm  = int( padding_width * scale_x)
            padding_height_hm = int( padding_height * scale_y)

        if self.debug:
            not_normalized_image = np.array(transformed_image.copy())
            # add keypoints to not_normalized_image
            # for i in range(0, len(keypoints), 2):
            #     x = int(keypoints[i].item())
            #     y = int(keypoints[i+1].item())
            #     cv2.circle(not_normalized_image, (x, y), 2, (255, 0, 0), -1)


        # Normalize image
        transformed_image = F.to_tensor(transformed_image)
        transformed_image = F.normalize(transformed_image, mean=[0.5] * 3, std=[0.5] * 3)

        """Original"""
        # # Calculate padding to match the aspect ratio
        # padding_width, padding_height = calculate_padding(*transformed_image.size, *self.output_size)

        # # print(f"padding width: {padding_width}, padding height: {padding_height}")
        # transformed_image = Pad(padding=(padding_width, padding_height), fill=0, padding_mode='constant')(transformed_image)
        # if not self.infer:
        #     # Add padding to keypoints
        #     keypoints += torch.tensor([padding_width, padding_height])  # Adjust for padding
        #     keypoints = keypoints.view(-1)
        # # Resize image to output size
        # # scale_x = self.output_size[1] / transformed_image.size[0]
        # # scale_y = self.output_size[0] / transformed_image.size[1]

        
        # scale_x = self.output_size[1] / transformed_image.size[0]
        # scale_y = self.output_size[0] / transformed_image.size[1]
        # transformed_image = F.resize(transformed_image, self.output_size)
        # if not self.infer:
        #     # Resize keypoints
        #     keypoints[::2] *= scale_x  # Scale x-coordinates
        #     keypoints[1::2] *= scale_y  # Scale y-coordinates
        #     padding_width_hm  = int( padding_width * scale_x)
        #     padding_height_hm = int( padding_height * scale_y)

        # if self.debug:
        #     not_normalized_image = np.array(transformed_image.copy())
        #     # # add keypoints to not_normalized_image
        #     # for i in range(0, len(keypoints), 2):
        #     #     x = int(keypoints[i].item())
        #     #     y = int(keypoints[i+1].item())
        #     #     cv2.circle(not_normalized_image, (x, y), 2, (255, 0, 0), -1)


        # # Normalize image
        # transformed_image = F.to_tensor(transformed_image)
        # transformed_image = F.normalize(transformed_image, mean=[0.5] * 3, std=[0.5] * 3)

        # ----------------------------------
        # ------- Generate heatmaps --------
        # ----------------------------------
        if not self.infer:
            heatmaps = []
            keypoints = keypoints.view(-1, 2)
            for keypoint in keypoints:
                heatmap = generate_heatmap(transformed_image, keypoint, 
                                            padding_width=padding_width_hm,
                                            padding_height=padding_height_hm,
                                            heatmap_size=(64, 48), sigma=0.8)
                heatmaps.append(heatmap)

            heatmaps = torch.stack(heatmaps)

        return transformed_image, keypoints, heatmaps, original_image, not_normalized_image

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

    # Normalize heatmap to range [0, 1]
    heatmap /= heatmap.max()

    # Convert to tensor
    return torch.tensor(heatmap, dtype=torch.float32)

if __name__ == "__main__":
    # Set paths
    image_folder = "test"
    label_file = "labels.csv"

    # Create dataset and data loader
    dataset = TopViewDataset(image_folder=image_folder,
                            label_file=label_file, 
                            output_size=(256, 192))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Iterate through the data loader
    # selec a random integer between 0 and dataset length
    index = random.randint(0, len(dataset))
    for i, (images, _, heatmaps, _, _) in enumerate(data_loader):
        if i % 5 == 0:  # Show every 10th batch
            image = images[0]
            heatmaps = heatmaps[0]

            overlap_hm = heatmaps[0]
            for hm in heatmaps[1:]:
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

