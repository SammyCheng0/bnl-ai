from marce.models import get_pose_net  # Make sure this import matches your setup

# Load the model architecture
model = get_pose_net(cfg, is_train=False)  # Ensure you're using the correct configuration

# Print the entire model to inspect the layers
print(model)
