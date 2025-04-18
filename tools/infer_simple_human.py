import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
from hrnet_w32_256 import get_pose_net
import seaborn as sns
import matplotlib.pyplot as plt

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load pre-trained HRNet-W32 model (e.g., from GitHub or TorchHub)
model = get_pose_net(is_train=True)

model.eval()


# Load the image
image_path = "data/Screenshot 2024-12-22 at 14.41.26.png"

# image_path = "path_to_your_image.jpg"  # Replace with your image path
original_image = Image.open(image_path).convert("RGB")

image = original_image.copy()

# 238 × 762

# 575 -> 287

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Pad(  # Add padding to make the image aspect ratio compatible
        padding=(
            184,  # Padding on left/right
            0
        ),
        fill=(0, 0, 0),  # Fill color for padding (black in this case)
        padding_mode='constant'
    ),
    transforms.Resize((256, 192)),  # Resize to 256x192
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

image = transform(image).unsqueeze(0)


# Perform inference with HRNet
with torch.no_grad():
    output = model(image)

# Process the output heatmaps to extract joint locations
heatmaps = output.squeeze().cpu().numpy()
# print(image.shape)
# exit()
_ , _ , height, width = image.shape
joints = []

print(heatmaps.shape)

for i in range(heatmaps.shape[0]):
    # print(heatmaps[i].shape)
    heatmap = heatmaps[i]
    # sns.heatmap(heatmap)
    # print(np.argmax(heatmap))
    # plt.show()
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    confidence = heatmap[y, x]
    # Scale joint locations to the original image size
    x = int(x / heatmap.shape[1] * width)
    y = int(y / heatmap.shape[0] * height)
    print(x, y, confidence)
    joints.append((x, y, confidence))

# Step 1: Remove the batch dimension (convert 4D -> 3D)
image_3d = image.squeeze(0)  # Shape: (3, 256, 256)

unnormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)
back_image = unnormalize(image_3d)  # Unnormalize the image

# plt.imshow(  image_3d.permute(1, 2, 0).numpy()  )
# plt.show()
# Step 2: Permute dimensions to match OpenCV's format (C, H, W -> H, W, C)
image_np = back_image.permute(1, 2, 0).numpy()  # Shape: (256, 192, 3)

# Step 3: Convert to uint8 (OpenCV expects pixel values in [0, 255])
image_np = (abs(image_np * 255)).astype(np.uint8)

image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# Debugging
print(type(image_np))  # Should be <class 'numpy.ndarray'>
print(image_np.dtype)  # Should typically be uint8
print(image_np.shape)  # Should be (height, width, channels)


# Draw the predicted joints on the image
for joint in joints:
    x, y, confidence = joint
    # if confidence > 0.5:  # Only plot joints with high confidence
    cv2.circle(image_np, (x, y), 3, (0, 255, 0), 2)

# Display the image with the predicted joints
cv2.imshow('Pose Estimation', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the result
# cv2.imwrite('output_pose_estimation.jpg', original_image)
