# import torch
# import torchvision.transforms as transforms
# import torchvision.models.segmentation as models
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

# # ---------------------------
# # Configuration
# # ---------------------------
# # Paths to your model checkpoint, test image, and ground truth mask
# model_path = "models/segmentation_model.pth"  # Trained model checkpoint (20 classes)
# test_image_path = "dataset/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
# gt_mask_path = "dataset/gtFine_trainvaltest/gtFine/test/berlin/berlin_000000_000019_gtFine_labelIds.png"

# # Set the number of classes to match your training (20 in this case)
# NUM_CLASSES = 20

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # ---------------------------
# # Model Initialization and Loading
# # ---------------------------
# # Initialize DeepLabV3 model with num_classes = 20
# model = models.deeplabv3_resnet101(pretrained=False, num_classes=NUM_CLASSES)
# model.to(device)

# # Load the checkpoint (ensure that the checkpoint was saved for a 20-class model)
# state_dict = torch.load(model_path, map_location=device)
# # If you get unexpected keys (e.g., from aux_classifier), you can remove them:
# # state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
# model.load_state_dict(state_dict, strict=True)
# model.eval()
# print("✅ Model loaded successfully!")
# print(model)

# # ---------------------------
# # Image & Mask Preprocessing
# # ---------------------------
# # Define transformation for the test image (must match training)
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize image to 256x256
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # Load and preprocess the test image
# image = Image.open(test_image_path).convert("RGB")
# input_image = transform(image).unsqueeze(0).to(device)

# # Load and resize ground truth mask to 256x256 using NEAREST interpolation
# gt_mask = Image.open(gt_mask_path)
# gt_mask = gt_mask.resize((256, 256), Image.NEAREST)
# gt_mask = np.array(gt_mask, dtype=np.int64)

# # ---------------------------
# # Inference
# # ---------------------------
# with torch.no_grad():
#     output = model(input_image)["out"]  # Output shape: [1, NUM_CLASSES, 256, 256]

# # Convert output logits to predicted segmentation mask by taking the argmax
# pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

# # ---------------------------
# # IoU Computation
# # ---------------------------
# def compute_iou(pred_mask, gt_mask, num_classes):
#     iou_scores = []
#     for cls in range(num_classes):
#         intersection = np.logical_and(pred_mask == cls, gt_mask == cls).sum()
#         union = np.logical_or(pred_mask == cls, gt_mask == cls).sum()
#         if union == 0:
#             continue  # Skip classes not present in the ground truth
#         iou_scores.append(intersection / union)
#     return np.mean(iou_scores) if iou_scores else 0.0

# mean_iou = compute_iou(pred_mask, gt_mask, NUM_CLASSES)
# print(f"Mean IoU Score: {mean_iou:.4f}")

# # ---------------------------
# # Visualization
# # ---------------------------
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# # Original image (resized)
# ax[0].imshow(image)
# ax[0].set_title("Original Image")
# ax[0].axis("off")

# # Ground truth mask (resized)
# ax[1].imshow(gt_mask, cmap="gray")
# ax[1].set_title("Ground Truth Mask")
# ax[1].axis("off")

# # Predicted segmentation mask
# ax[2].imshow(pred_mask, cmap="jet")
# ax[2].set_title("Predicted Segmentation Mask")
# ax[2].axis("off")

# plt.show()
import torch
import torchvision.models.segmentation as models

# Define model with the correct number of classes
num_classes = 21  # Set this to your actual dataset classes
model = models.deeplabv3_resnet101(pretrained=False, aux_loss=True)

# Load the checkpoint but ignore classifier weights
checkpoint_path = "/home/harishankar/CNN_Model/models/segmentation_model.pth"
state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# Remove incompatible keys
del state_dict["classifier.4.weight"]
del state_dict["classifier.4.bias"]

# Load remaining weights
model.load_state_dict(state_dict, strict=False)

# Reinitialize the classifier layer for 21 classes
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
torch.nn.init.xavier_uniform_(model.classifier[4].weight)
torch.nn.init.zeros_(model.classifier[4].bias)

print("Model loaded with modified classifier!")


