import csv
import time
import torch
import signal
import torch.nn as nn
import torch.optim as optim
import platform
import multiprocessing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from hrnet_w32_256 import get_pose_net 
from SideViewDataset import SideViewDataset
import os
from datetime import datetime

if __name__ == '__main__':

    if platform.system() == "Darwin":  # "Darwin" is the name for macOS
        multiprocessing.set_start_method("fork", force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Torch version: {torch.__version__}')
    print(f'Torch device: {device}')

    # if device.type == 'cpu':
    #     torch.set_num_threads(torch.get_num_threads())
    #     torch.set_num_interop_threads(torch.get_num_interop_threads())

    # def signal_handler(sig, frame):
    #     print('Saving loss graph...')
    #     plt.figure()
    #     plt.plot(train_losses)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Train Loss')
    #     plt.savefig(os.path.join(output_folder, 'train_loss.png'))
    #     exit(0)

    # signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    # signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals


    output_folder = f'out/train-{datetime.now().strftime("%y%m%d_%H%M%S")}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    learning_rate = 0.001
    epochs = 1000
    patience = 10
    lowest_test_loss = float('inf')

    # train_dataset = TopViewDataset(image_folder='data/dataset/train', 
    #                             label_file='data/dataset/labels.csv', 
    #                             output_size=(256, 192))
    # test_dataset  = TopViewDataset(image_folder='data/dataset/test',
    #                             label_file='data/dataset/labels.csv',
    #                             output_size=(256, 192))

    train_dataset = SideViewDataset(image_folder='train', 
                                label_file='sorted_annotations.csv', 
                                output_size=(256, 192))
    test_dataset  = SideViewDataset(image_folder='test',
                                label_file='sorted_annotations.csv',
                                output_size=(256, 192))


    train_batch_size = 10
    test_batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=1, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=test_batch_size,  num_workers=1, shuffle=False)

    model = get_pose_net(is_train=False).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    count_patience = 0
    for epoch in range(1, epochs + 1):
        if count_patience >= patience:
            print(f'Early stopping at epoch {epoch}...')
            break
        count_patience += 1
        model.train()
        train_loss = 0.0
        start_time = time.time()
        num_batches = 0
        for batch_idx, (images, _, heatmaps, _, _) in enumerate(train_dataloader):
            images, heatmaps = images.to(device), heatmaps.to(device)
            num_batches += 1
            optimizer.zero_grad()
            prediction = model(images)

            loss = criterion(prediction, heatmaps)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f'[{(batch_idx + 1) * train_batch_size} / {len(train_dataset)}]: loss: {loss.item()}')
        train_loss /= num_batches

        model.eval()
        test_loss = 0.0
        num_batches = 0
        for batch_idx, (images, _, heatmaps, _, _) in enumerate(test_dataloader):
            images, heatmaps = images.to(device), heatmaps.to(device)
            num_batches += 1
            prediction = model(images)
            loss = criterion(prediction, heatmaps)
            test_loss += loss.item()
            # print(f'{batch_idx}: loss: {loss.item()}')
        test_loss /= num_batches
        overall_time = time.time() - start_time

        print(f"Epoch [{epoch}/{epochs}], Average Train Loss: {train_loss}, Average Test Loss: {test_loss}" \
              f", Time: {overall_time:.2f}")

        if epoch % 100 == 0:
            print(f'Saving snapshot at epoch {epoch}...')
            torch.save(model.state_dict(), os.path.join(output_folder, f'snapshot_{epoch}.pth'))

        with open(os.path.join(output_folder, 'loss.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:  # Check if the file is empty
                writer.writerow(['epoch', 'average_train_loss', 'average_test_loss', 'overall_time'])
            writer.writerow([epoch, train_loss, test_loss, overall_time])

        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
            count_patience = 0
            print(f'Saving best model...')
            torch.save(model.state_dict(), os.path.join(output_folder, 'snapshot_best.pth'))
