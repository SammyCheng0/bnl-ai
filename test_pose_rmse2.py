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
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    model.eval()
    model_name = getattr(model, "name", model.__class__.__name__)

    output_folder = f'out/test-{datetime.now().strftime("%y%m%d_%H%M%S")}-{model_name}'
    image_folder = os.path.join(output_folder, 'images')
    os.makedirs(image_folder, exist_ok=True)

    # Load and filter annotation columns
    df = pd.read_csv(annotation_path)
    df = df.drop(columns=[col for col in df.columns if col.startswith("bbox_")], errors='ignore')
    keypoint_columns = [col for col in df.columns if col.endswith('-x') and col[:-2] + '-y' in df.columns]
    joint_names = [col[:-2] for col in keypoint_columns]

    test_dataset = PoseDataset(image_folder=image_test_folder,
                               label_file=annotation_path,
                               resize_to=input_size,
                               heatmap_size=output_size)
    dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

    info_file = os.path.join(output_folder, 'info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Model name: {model_name}\n")
        f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"Test dataset: {image_test_folder}\n")
        f.write(f"Annotation path: {annotation_path}\n")
        f.write(f"Input size: {input_size}\n")
        f.write(f"Output size: {output_size}\n")
        f.write(f"Batch size: 1\n")

    def extract_keypoints_with_confidence(heatmaps):
        keypoints = []
        for heatmap in heatmaps:
            heatmap = heatmap.squeeze(0)
            max_val, max_idx = torch.max(heatmap.view(-1), dim=0)
            y, x = divmod(max_idx.item(), heatmap.size(1))
            keypoints.append(((x, y), max_val.item()))
        return keypoints

    confidence_thresh = 0.1
    list_SE = []
    per_image_rmse = []

    with torch.no_grad():
        for batch_idx, (images, gt_keypoints, gt_hms, scaler_kps) in enumerate(dataloader):
            filename = test_dataset.image_filenames[batch_idx]

            images = images.to(device)
            predictions = model(images).squeeze(0)
            image = images.squeeze(0)
            gt_keypoints = gt_keypoints.squeeze(0)
            scaler_kps = scaler_kps.squeeze(0)

            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), image.shape[1:]) for hm in predictions])
            keypoints_with_conf = extract_keypoints_with_confidence(resized_heatmaps)
            keypoints = torch.tensor([pt for pt, conf in keypoints_with_conf], dtype=torch.float32)
            confidences = torch.tensor([conf for pt, conf in keypoints_with_conf])

            if confidences.le(confidence_thresh).all():
                print(f"Skipping image {batch_idx} — no confident keypoints")
                continue

            squared_error = torch.sum((keypoints / scaler_kps - gt_keypoints / scaler_kps) ** 2, dim=1)
            squared_error[confidences <= confidence_thresh] = torch.nan
            list_SE.append(squared_error)

            rmse = torch.sqrt(torch.nanmean(squared_error))
            per_image_rmse.append((filename, rmse))
            
            # Denormalize the image
            mean, std = load_mean_std_from_file(r'sammy\train_normalization.txt', device=device)
            denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
            denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
            denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range
            denormalized_image = np.transpose(denormalized_image, (1, 2, 0))  # Change shape to HWC for plotting

            # Convert list of tuples to separate x and y coordinate lists
            keypoints_with_conf_x = [kp[0] for kp in keypoints_with_conf]
            keypoints_with_conf_y = [kp[1] for kp in keypoints_with_conf]

            # # Plot the image with keypoints
            # plt.figure(figsize=(6, 6))
            # plt.imshow(denormalized_image)
            # plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='green', marker='o', label="Ground Truth")  # Ground truth in green
            # plt.scatter(keypoints_with_conf_x, keypoints_with_conf_y, c='red', marker='x', label="Predicted")  # Predictions in red
            # plt.legend()
            # plt.axis("on")
            # plt.axis("equal")

            # # Save the image to the output folder
            # figure_path = os.path.join(image_folder, f"{batch_idx}.png")  # Save as PNG with the batch index as filename
            # plt.savefig(figure_path, bbox_inches='tight', pad_inches=0, dpi=300)
            # plt.close()  # Close the figure to free up memory
            
            mean, std = load_mean_std_from_file(r'sammy\train_normalization.txt', device=device)
            
            denormalized_image = (image * std + mean).to('cpu')  # Reverse normalization
            denormalized_image = denormalized_image.clamp(0, 1)  # Ensure valid range
            denormalized_image = (denormalized_image * 255).byte().numpy()  # Convert to 0-255 range
            denormalized_image = np.transpose(denormalized_image, (1, 2, 0))
            
            # Extract x and y coordinates from keypoints_with_conf
            keypoints_with_conf_x = [kp[0][0] for kp in keypoints_with_conf]  # x-coordinates of predicted keypoints
            keypoints_with_conf_y = [kp[0][1] for kp in keypoints_with_conf]  # y-coordinates of predicted keypoints

            # Plot the image
            plt.figure(figsize=(6, 6))
            plt.imshow(denormalized_image)
            plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='green', marker='o', label="Ground Truth")  # Ground truth in green
            plt.scatter(keypoints_with_conf_x, keypoints_with_conf_y, c='red', marker='x', label="Predicted")  # Predictions in red
            plt.legend()
            plt.axis("on")
            plt.axis("equal")
            figure_path = os.path.join(image_folder, str(batch_idx))
            plt.savefig(figure_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

            # # Plot the image
            # plt.figure(figsize=(6, 6))
            # plt.imshow(denormalized_image)
            # plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='green', marker='o', label="Ground Truth")  # Ground truth in green
            # plt.scatter(keypoints_with_conf[:, 0], keypoints_with_conf[:, 1], c='red', marker='x', label="Predicted")  # Predictions in red
            # plt.legend()
            # plt.axis("on")
            # plt.axis("equal")
            # figure_path = os.path.join(image_folder, str(batch_idx))
            # plt.savefig(figure_path, bbox_inches='tight', pad_inches=0, dpi=300)
            # plt.close()


    # Stack squared errors
    stacked_SE = torch.stack(list_SE)
    rmse_all = torch.sqrt(torch.nanmean(stacked_SE))
    rmse_std = np.sqrt(np.nanstd(stacked_SE.cpu().numpy()))
    rmse_per_joint = torch.sqrt(torch.nanmean(stacked_SE, dim=0))

    # Count valid predictions (after applying confidence threshold)
    valid_counts = torch.zeros(len(joint_names), dtype=torch.int32)
    with torch.no_grad():
        for batch_idx, (images, gt_keypoints, gt_hms, scaler_kps) in enumerate(dataloader):
            images = images.to(device)
            predictions = model(images).squeeze(0)
            image = images.squeeze(0)
            scaler_kps = scaler_kps.squeeze(0)

            resized_heatmaps = torch.stack([resize(hm.unsqueeze(0), image.shape[1:]) for hm in predictions])
            keypoints_with_conf = extract_keypoints_with_confidence(resized_heatmaps)
            confidences = torch.tensor([conf for pt, conf in keypoints_with_conf])

            valid_mask = confidences > confidence_thresh
            valid_counts += valid_mask.int()

    # Count how many times each ground truth keypoint is present (not NaN)
    gt_counts = torch.zeros(len(joint_names), dtype=torch.int32)
    for _, gt_keypoints, _, scaler_kps in dataloader:
        gt_keypoints = gt_keypoints.squeeze(0)
        gt_counts += ~torch.isnan(gt_keypoints[:, 0])  # x values

    with open(info_file, 'a') as f:
        f.write(f"\n------------------------\n")
        f.write(f"Overall average RMSE      : {rmse_all.item():.4f}\n")
        f.write(f"Standard deviation of RMSE: {rmse_std:.4f}\n")
        f.write(f"\nAverage RMSE for each test image:\n")
        for fname, rmse in per_image_rmse:
            f.write(f"  {fname:30s}: RMSE = {rmse.item():.4f}\n")

        f.write("\nRMSE per joint and prediction counts:\n")
        for name, rmse, pred_count, gt_count in zip(joint_names, rmse_per_joint, valid_counts, gt_counts):
            f.write(f"  {name:25s}: RMSE = {rmse.item():.4f}, Predicted = {int(pred_count)}, GT = {int(gt_count)}\n")

    return rmse_all.item(), rmse_std


if __name__ == '__main__':
    image_test_folder  = r'sammy\sideview\test'
    annotation_path    = r'sammy\sideview\merged_labels.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    model = SHG.get_pose_net(nstack=2)
    model = model.to(device)
    
    # Load the checkpoint and handle potential errors
    try:
        # Load the model's state dictionary (weights)
        checkpoint = torch.load(r"out\train-250429_145152_32\snapshot_best.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=False)  # Use strict=False to ignore missing/extra keys
        print(f"Model weights loaded successfully from {'out\train-250429_145152_32\snapshot_best.pth'}")

    except RuntimeError as e:
        print(f"Error loading model: {e}")
        raise e

    with open(r'sammy\data\hrnet_w32_384_288.yaml', 'r') as f:
        cfg_w32_256_192 = yaml.load(f, Loader=yaml.SafeLoader)
        cfg_w32_256_192['MODEL']['NUM_JOINTS'] = 26
        model = hrnet.get_pose_net(cfg_w32_256_192, is_train=False)
        model = model.to(device)
        model.load_state_dict(torch.load(r"out\train-250429_145152_32\snapshot_best.pth", weights_only=True, map_location=device))
        test_pose(model, image_test_folder, annotation_path, 
                  input_size=cfg_w32_256_192['MODEL']['IMAGE_SIZE'],
                  output_size=cfg_w32_256_192['MODEL']['HEATMAP_SIZE'])
