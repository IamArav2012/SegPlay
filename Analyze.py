import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Lung_Segmentator as ls
import os
from tensorflow.keras.utils import load_img, img_to_array

def load_and_preprocess_samples(num_total_samples):
    lung_images = []
    lung_masks = []

    if num_total_samples // 3 != num_total_samples / 3:
        fixed_samples = (num_total_samples // 3) * 3
        print(f"Adjusted num_total_samples from {num_total_samples} → {fixed_samples} (must be multiple of 3)")
        num_total_samples = fixed_samples

    samples_per_dataset = num_total_samples // 3

    root_folders = [f for f in os.listdir(base_dir) if f != 'utis']
    if len(root_folders) < 3:
        raise ValueError("Less than 3 valid dataset folders found.")

    for root_folder in root_folders:
        root_path = os.path.join(base_dir, root_folder)
        img_dir = os.path.join(root_path, 'img')
        mask_dir = os.path.join(root_path, 'mask')

        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue

        img_files = sorted(os.listdir(img_dir))
        count = 0

        for img_file in img_files:
            if count >= samples_per_dataset:
                break

            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)

            if not os.path.exists(mask_path):
                continue

            img = img_to_array(load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb'))
            mask = img_to_array(load_img(mask_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale'))

            lung_images.append(img)
            lung_masks.append(mask)
            count += 1
            print(f"Loaded: {len(lung_images)}")

    lung_images = np.array(lung_images).astype(np.float32) / 255.0
    lung_masks = (np.array(lung_masks) > 0.5).astype(np.float32)

    lung_images = lung_images.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    lung_masks = lung_masks.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    dataset = tf.data.Dataset.from_tensor_slices((lung_images, lung_masks)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def sample_from_dataset(dataset, num_samples_to_get):
    collected_images = []
    collected_masks = []
    samples_collected = 0

    if num_samples_to_get > len(dataset):
        raise ValueError(f'''Dataset is smaller than the amount of samples requested ({num_samples_to_get} samples). 
                         Maximum samples for this dataset is {len(dataset)}''')

    for batch_images, batch_masks in dataset:
        batch_size = tf.shape(batch_images)[0].numpy()

        samples_needed_from_batch = min(batch_size, num_samples_to_get - samples_collected)

        if samples_needed_from_batch <= 0:
            break

        for i in range(samples_needed_from_batch):
            collected_images.append(batch_images[i])
            collected_masks.append(batch_masks[i])
            samples_collected += 1

        if samples_collected >= num_samples_to_get:
            break

    if not collected_images:
        final_images = np.array([])
        final_masks = np.array([])
        print(f"Warning: No samples were collected.")
    else:
        final_images = np.array(collected_images)
        final_masks = np.array(collected_masks)

    print(f"Collected {samples_collected} samples.")
    print(f"Final images shape: {final_images.shape}")
    print(f"Final masks shape: {final_masks.shape}")

    return final_images, final_masks

def manual_evaluate(model, dataset, threshold=0.5):

    y_preds = []
    y_trues = []

    for x_batch, y_batch in dataset:
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_preds.append(y_pred_batch)
        y_trues.append(y_batch)

    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)

    y_preds_bin = (y_preds > threshold).astype(np.float32)

    def dice_coef_np(y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        return (2. * intersection + 1e-7) / (union + 1e-7)

    def iou_np(y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return (intersection + 1e-7) / (union + 1e-7)

    def pixel_accuracy_np(y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        total = np.prod(y_true.shape)
        return correct / total

    bce_fn = tf.keras.losses.BinaryCrossentropy()
    bce_loss = bce_fn(tf.convert_to_tensor(y_trues), tf.convert_to_tensor(y_preds)).numpy()

    dice_loss_val = 1 - dice_coef_np(y_trues, y_preds_bin)

    return {
        "Dice Coefficient": dice_coef_np(y_trues, y_preds_bin),
        "Dice Loss": dice_loss_val,
        "IoU": iou_np(y_trues, y_preds_bin),
        "Pixel Accuracy": pixel_accuracy_np(y_trues, y_preds_bin),
        "Binary Crossentropy": bce_loss
    }

base_dir = r'D:\ML\Medical Datasets\Chest X-ray dataset for lung segmentation' 
IMG_SIZE = 128
batch_size = 4

model = tf.keras.models.load_model('lung_mri_segmentator.keras', 
                                   custom_objects={
                                    'dice_coef': ls.dice_coef,
                                    'combined_loss': ls.combined_loss,
                                   })

# Extract test dataset
test_dataset = load_and_preprocess_samples(500)

metrics = manual_evaluate(model, test_dataset)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Load individual images and masks
images, masks = sample_from_dataset(test_dataset, num_samples_to_get=10)
y_pred = model.predict(images)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(images[0], cmap='gray')
plt.title('Lung Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(y_pred[0, ..., 0], cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(masks[0, ..., 0], cmap='gray')
plt.title('Actual Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

def overlay_mask_on_image(image, mask, alpha=0.5, mask_color=[1, 0, 0]):

    img = image.copy()
    if img.max() > 1:
        img = img / 255.0

    color_mask = np.zeros_like(img)
    for i in range(3):
        color_mask[..., i] = mask * mask_color[i]

    overlayed = img * (1 - alpha) + color_mask * alpha
    overlayed = np.clip(overlayed, 0, 1)

    return overlayed

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(images[1])
plt.axis('off')

plt.subplot(2,2,2)
plt.title("True Mask")
plt.imshow(masks[1,...,0], cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Predicted Mask")
plt.imshow( y_pred[1,...,0], cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Overlayed")
plt.imshow(overlay_mask_on_image(images[1], y_pred[1,...,0]))
plt.axis('off')

plt.show()