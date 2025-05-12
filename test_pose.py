import csv
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PoseDataset import PoseDataset
import hrnet
# import resnet
import SHG
from torchvision.transforms.functional import resize
import os
from datetime import datetime
import pandas as pd

def load_mean_std_from_file(file_path, device='cpu'):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    def parse_tensor_line(line):
        tensor_str = line.split("tensor(")[-1].rstrip(")\n")
        numbers = [float(x.strip()) for x in tensor_str.strip("[]").split(',')]
        return torch.tensor(numbers).view(3, 1, 1).to(device)

    mean = parse_tensor_line(lines[0])
    std = parse_tensor_line(lines[1])
    return mean, std

def test_pose(model, image_test_folder, annotation_path, input_size, output_size):

    if platform.system() == "Darwin":  # "Darwin" is the name for macOS
        multiprocessing.set_start_method("fork", force=True)

    model.eval()

    # if model.name:
    #     model_name = model.name
    # else:
    #     model_name = model.__class__.__name__
    
    model_name = getattr(model, "name", model.__class__.__name__)

    output_folder = f'out/test-{datetime.now().strftime("%y%m%d_%H%M%S")}-{model_name}'
    image_folder = os.path.join(output_folder, 'images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    test_dataset  = PoseDataset(image_folder=image_test_folder,
                                label_file=annotation_path,
                                resize_to=input_size,
                                heatmap_size=output_size)
    test_batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=1, shuffle=False)

    info_file = os.path.join(output_folder, 'info.txt')
    with open(info_file, 'w') as file:
        file.write(f"Model name: {model_name}\n")
        file.write(f"number parameters: {sum(p.numel() for p in model.parameters())}\n")
        file.write(f"image_test_folder: {image_test_folder}\n")
        file.write(f"annotation_path: {annotation_path}\n")
        file.write(f"input_size: {input_size}\n")
        file.write(f"output_size: {output_size}\n")
        file.write(f"test_batch_size: {test_batch_size}\n")
        # file.write(f"keypoint_names: {test_dataset.keypoint_names}\n")

    def extract_keypoints_with_confidence(heatmaps):
        """
        Extracts keypoints and their confidence from heatmaps.
        Args:
            heatmaps: Tensor of shape (num_keypoints, h, w) containing heatmaps for each keypoint.
        Returns:
            keypoints_with_confidence: List of ((x, y), confidence) for each keypoint.
        """
        keypoints_with_confidence = []
        for heatmap in heatmaps:
            heatmap = heatmap.squeeze(0)
            # Get the maximum value and its index
            max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
            y, x = divmod(max_idx.item(), heatmap.size(1))  # Convert linear index to x, y
            
            # Confidence as the maximum value
            confidence = max_val.item()
            keypoints_with_confidence.append(((x, y), confidence))
        return keypoints_with_confidence

    confidence = 0.02
    with torch.no_grad():
        test_loss = 0.0
        num_batches = 0
        pck = []
        list_SE = []
        for batch_idx, (images, gt_keypoints, gt_hms, scaler_kps) in enumerate(dataloader):
            num_batches += 1
            images = images.to(device)
            start = time.time()
            predictions = model(images)

            end = time.time()
            print(f"Model output channels: {predictions.size(1)}")
            # print(f"Image {batch_idx} took {end - start:.4f} seconds")
            image = images.squeeze(0)
            """_summary_
            """            
            gt_keypoints = gt_keypoints.squeeze(0)
            gt_hms = gt_hms.squeeze(0)
            predictions = predictions.squeeze(0)
            original_size = image.shape[1:]  # Assume CHW format

            # caclulating keypoints from heatmaps
            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), original_size, ) for hm in predictions])
            keypoints = extract_keypoints_with_confidence(resized_heatmaps)
            keypoints_without_confidence = torch.tensor([t[0] for t in keypoints], dtype=torch.float32)
            keypoints_with_confidence = torch.tensor([t[0] for t in keypoints if t[1] > confidence], dtype=torch.float32)
            
            # 👉 Add this here
            if keypoints_with_confidence.numel() == 0:
                print(f"Skipping image {batch_idx} — no confident keypoints found (threshold={confidence})")
                continue  # Skip to the next image
            
            print(f"Predicted keypoints: {keypoints_without_confidence.size()}")
            print(f"Ground truth keypoints: {gt_keypoints.size()}")


            list_SE.append(torch.sum((keypoints_without_confidence/scaler_kps - gt_keypoints/scaler_kps) ** 2, dim=1))


            # Denormalize the image
            # mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            # std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            
            mean, std = load_mean_std_from_file(r'sammy\train_normalization.txt', device=device)
            
            denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
            denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
            denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range
            denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

            # Plot the image
            plt.figure(figsize=(6, 6))
            plt.imshow(denormalized_image)
            plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='green', marker='o', label="Ground Truth")  # Ground truth in green
            plt.scatter(keypoints_with_confidence[:, 0], keypoints_with_confidence[:, 1], c='red', marker='x', label="Predicted")  # Predictions in red
            plt.legend()
            plt.axis("on")
            plt.axis("equal")
            figure_path = os.path.join(image_folder, str(batch_idx))
            plt.savefig(figure_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            # plt.show()
            # print(f"Image saved at {figure_path}")
        stacked_SE = torch.stack(list_SE, dim=0)
        RMSE = torch.sqrt(torch.nanmean(stacked_SE))
        # if model.name:
        #     print(f"{model.name}-{input_size}, RMSE = {RMSE}")
        # else:
        #     print(f"{model. __class__. __name__}-{input_size}, RMSE = {RMSE}")
        model_name = getattr(model, "name", model.__class__.__name__)
        print(f"{model_name}-{input_size}, RMSE = {RMSE}")


    with open(info_file, 'a') as file:
        file.write(f"\n ------------------------ \n\n")
        file.write(f"RMSE: {RMSE}\n")

if __name__ == '__main__':
    image_test_folder  = r'sammy\sideview\test'
    annotation_path    = r'sammy\sideview\merged_labels.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    # model = SHG.get_pose_net(nstack=2)
    # model = model.to(device)
    # model.load_state_dict(torch.load(r"out\train-250411_125855-PoseHRNet-W48\snapshot_best.pth", weights_only=True, map_location=device))
    # test_pose(model, image_test_folder, annotation_path, 
    #           input_size=[256, 256],
    #           output_size=[64, 64])
    
    model = SHG.get_pose_net(nstack=2)
    model = model.to(device)
    
    # Load the checkpoint and handle potential errors
    try:
        # Load the model's state dictionary (weights)
        checkpoint = torch.load(r"out\train-250429_145152\snapshot_best.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=False)  # Use strict=False to ignore missing/extra keys
        print(f"Model weights loaded successfully from {'out\train-250429_145152\snapshot_best.pth'}")

    except RuntimeError as e:
        print(f"Error loading model: {e}")
        # You could add additional error handling if necessary, or re-raise the error.
        raise e

    with open(r'sammy\data\hrnet_w48_384_288.yaml', 'r') as f:
            cfg_w32_256_192 = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_w32_256_192['MODEL']['NUM_JOINTS'] = 26
            model = hrnet.get_pose_net(cfg_w32_256_192, is_train=False)
            model = model.to(device)
            model.load_state_dict(torch.load(r"out\train-250429_145152\snapshot_best.pth", weights_only=True, map_location=device))
            test_pose(model, image_test_folder, annotation_path, 
                      input_size=cfg_w32_256_192['MODEL']['IMAGE_SIZE'],
                      output_size=cfg_w32_256_192['MODEL']['HEATMAP_SIZE'])
   