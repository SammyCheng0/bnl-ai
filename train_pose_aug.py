import argparse
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset, random_split
from PoseDataset import PoseDataset
import hrnet
import resnet
import SHG 
import sys
from alive_progress import alive_bar
# from alive_progress.animations import bar_factory
from tools.plot_losses import plot_losses
from test_pose import test_pose
import yaml
from datetime import datetime
import os
import mlflow
import optuna

def train_pose(model, image_train_folder, image_val_folder, 
               annotation_path, input_size, output_size, device='cpu',
               n_joints=None, train_batch_size=25, patience=10, 
               start_lr=0.001, min_lr=0.00001, epochs=10000,
               threshold=1e-3, augmentations=list(), output_folder=None, trial=None):

    model_name = model.__class__.__name__
    if not output_folder:
        output_folder = f'out/train-{datetime.now().strftime("%y%m%d_%H%M%S")}-{model_name}'
        os.makedirs(output_folder, exist_ok=True)

    mlflow.log_params({
        "model_name": model_name,
        "epochs": epochs,
        "learning_rate": start_lr,
        "batch_size": train_batch_size,
        "patience": patience,
        "loss_function": "MSELoss",
        "optimizer": "Adam",
        "augmentations": ','.join(augmentations)
    })

    aug_flags = {aug: (aug in augmentations) for aug in ["rotate", "scale", "motion_blur", "brightness", "contrast", "sharpness", "gamma"]}

    train_dataset = PoseDataset(image_folder=image_train_folder,
                                label_file=annotation_path,
                                resize_to=input_size,
                                heatmap_size=output_size,
                                **aug_flags)
    
    val_dataset = PoseDataset(image_folder=image_val_folder,
                              label_file=annotation_path,
                              resize_to=input_size,
                              heatmap_size=output_size,
                              rotate=False, scale=False, motion_blur=False,
                              brightness=False, contrast=False, sharpness=False, gamma=False)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience, threshold=threshold, min_lr=min_lr)

    loss_csv_path = os.path.join(output_folder, f'loss_{model_name}.csv')
    lowest_val_loss = float('inf')
    total_time = 0.0
    prev_lr = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        num_batches = 0

        # bar = bar_factory('.', tip='🚀', background=' ', borders=('🌒','🌌'))
        # with alive_bar(len(train_loader.dataset), title=f"Epoch [{epoch}/{epochs}]", bar=bar, spinner='dots') as bar:
        with alive_bar(len(train_loader.dataset), title=f"Epoch [{epoch}/{epochs}]", bar="smooth", spinner='dots') as bar:
            for images, _, gt_hms, _ in train_loader:
                images, gt_hms = images.to(device), gt_hms.to(device)
                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, gt_hms)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
                bar.text = f"Train Loss: {loss.item():.10f}"
                bar(images.shape[0] if (num_batches * train_batch_size > len(train_loader.dataset)) else train_batch_size)

        train_loss /= num_batches

        model.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for images, _, gt_hms, _ in val_loader:
                images, gt_hms = images.to(device), gt_hms.to(device)
                preds = model(images)
                loss = criterion(preds, gt_hms)
                val_loss += loss.item()
                num_batches += 1
        val_loss /= num_batches

        elapsed = time.time() - start_time
        total_time += elapsed

        # Erase the progress bar line from terminal
        sys.stdout.write('\033[A\033[K')

        # Step LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if prev_lr == -1:
            prev_lr = current_lr
        elif current_lr != prev_lr:
            print(f"Learning rate changed to {current_lr:.2e}")
            prev_lr = current_lr

        print(f"Epoch [{epoch}/{epochs}] | Avg. Train Loss: {train_loss:.5e} | Avg. Val Loss: {val_loss:.5e} | Time: {elapsed:.2f}s")

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        with open(loss_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["epoch", "train_loss", "val_loss", "time"])
            writer.writerow([epoch, train_loss, val_loss, elapsed])

        if val_loss < lowest_val_loss - threshold:
            lowest_val_loss = val_loss
            print(f"Saving as best model...")
            torch.save(model.state_dict(), os.path.join(output_folder, 'snapshot_best.pth'))

        if current_lr <= min_lr:
            print(f"Stopping training: learning rate has reached the minimum threshold ({min_lr})")
            break

        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    plot_losses(loss_csv_path)
    return lowest_val_loss, output_folder

# def train_pose(model, image_train_folder, image_val_folder, 
#                annotation_path, input_size, output_size, device='cpu',
#                n_joints=None, train_batch_size=25, patience=10, 
#                start_lr=0.001, min_lr=0.00001, epochs=10000,
#                threshold=1e-3, augmentations=list(), output_folder=None, trial=None):

#     model_name = model.__class__.__name__
#     if not output_folder:
#         output_folder = f'out/train-{datetime.now().strftime("%y%m%d_%H%M%S")}-{model_name}'
#         os.makedirs(output_folder, exist_ok=True)

#     mlflow.log_params({
#         "model_name": model_name,
#         "epochs": epochs,
#         "learning_rate": start_lr,
#         "batch_size": train_batch_size,
#         "patience": patience,
#         "loss_function": "MSELoss",
#         "optimizer": "Adam",
#         "augmentations": ','.join(augmentations)
#     })

#     aug_flags = {aug: (aug in augmentations) for aug in ["rotate", "scale", "motion_blur", "brightness", "contrast", "sharpness", "gamma"]}

#     train_dataset = PoseDataset(image_folder=image_train_folder,
#                                 label_file=annotation_path,
#                                 resize_to=input_size,
#                                 heatmap_size=output_size,
#                                 **aug_flags)
    
#     val_dataset = PoseDataset(image_folder=image_val_folder,
#                               label_file=annotation_path,
#                               resize_to=input_size,
#                               heatmap_size=output_size,
#                               rotate=False, scale=False, motion_blur=False,
#                               brightness=False, contrast=False, sharpness=False, gamma=False)

#     train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
#     print("check train")

#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

#     print("check val")

#     # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
#     # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=start_lr)
#     criterion = nn.MSELoss()
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience, threshold=threshold, min_lr=min_lr)

#     loss_csv_path = os.path.join(output_folder, f'loss_{model_name}.csv')
#     lowest_val_loss = float('inf')
#     total_time = 0.0

#     for epoch in range(1, epochs + 1):
#         model.train()
#         train_loss = 0.0
#         start_time = time.time()

#         print(">>> Starting training loop")
#         for images, _, gt_hms, _ in train_loader:
#             images, gt_hms = images.to(device), gt_hms.to(device)
#             optimizer.zero_grad()
#             preds = model(images)
#             loss = criterion(preds, gt_hms)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)

#         model.eval()
#         val_loss = 0.0
#         print(">>> Starting validation loop")
#         with torch.no_grad():
#             for images, _, gt_hms, _ in val_loader:
#                 images, gt_hms = images.to(device), gt_hms.to(device)
#                 preds = model(images)
#                 loss = criterion(preds, gt_hms)
#                 val_loss += loss.item()
#         val_loss /= len(val_loader)
#         elapsed = time.time() - start_time
#         total_time += elapsed

#         mlflow.log_metric("train_loss", train_loss, step=epoch)
#         mlflow.log_metric("val_loss", val_loss, step=epoch)

#         with open(loss_csv_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             if f.tell() == 0:
#                 writer.writerow(["epoch", "train_loss", "val_loss", "time"])
#             writer.writerow([epoch, train_loss, val_loss, elapsed])

#         if val_loss < lowest_val_loss - threshold:
#             lowest_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(output_folder, 'snapshot_best.pth'))

#         scheduler.step(val_loss)
#         if optimizer.param_groups[0]['lr'] <= min_lr:
#             break

#         if trial:
#             trial.report(val_loss, epoch)
#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()

#     plot_losses(loss_csv_path)
#     return lowest_val_loss, output_folder

def objective(trial, args):
    selected_augs = []
    if trial.number < len(args.augmentations):
        selected_augs = [args.augmentations[trial.number]]
    else:
        for aug in args.augmentations:
            if trial.suggest_categorical(aug, [True, False]):
                selected_augs.append(aug)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['NUM_JOINTS'] = args.n_joints
    cfg['DEVICE'] = args.device
    model = hrnet.get_pose_net(cfg, is_train=True)

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params({
            "trial_number": trial.number,
            "augmentation_combo": ','.join(selected_augs)
        })

        best_val_loss, output_folder = train_pose(
            model=model,
            image_train_folder=args.image_train_folder,
            image_val_folder=args.image_val_folder,
            annotation_path=args.annotation_path,
            input_size=cfg['MODEL']['IMAGE_SIZE'],
            output_size=cfg['MODEL']['HEATMAP_SIZE'],
            n_joints=cfg['MODEL']['NUM_JOINTS'],
            train_batch_size=args.batch_size,
            patience=args.patience,
            start_lr=args.lr,
            min_lr=args.min_lr,
            epochs=args.epochs,
            threshold=args.threshold,
            augmentations=selected_augs,
            output_folder=None,
            device=args.device,
            trial=trial
        )
        mlflow.log_metric("final_val_loss", best_val_loss)
        return best_val_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_train_folder', type=str, required=True)
    parser.add_argument('--image_val_folder', type=str, required=True)
    parser.add_argument('--image_test_folder', type=str, required=True)
    parser.add_argument('--annotation_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/hrnet_w32_256_192.yaml')
    parser.add_argument('--n_joints', type=int, default=26)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--augmentations', nargs='+', default=['rotate', 'scale', 'motion_blur', 'brightness', 'contrast', 'sharpness', 'gamma'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_testing_after_training', action='store_true')
    parser.add_argument('--type', type=str, default='side')  # used in test path
    parser.add_argument('--start_lr', type=float, default=0.001, help='Starting learning rate')
    parser.add_argument('--train_batch_size', type=int, default=50, help='Training batch size')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=5)

    print("Best trial:")
    print(study.best_trial)

