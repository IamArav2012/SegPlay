import tensorflow as tf
import numpy as np
from pretraining import Patchify, PatchEncoder
from config import fine_tuned_model, batch_size, dice_loss, dice_coef, combined_loss, save_datasets, new_directory_name_for_npy_files, parse_image
import matplotlib.pyplot as plt
import os

def sample_from_dataset(dataset, num_samples_to_get):
    collected_images = []
    collected_masks = []
    samples_collected = 0

    if num_samples_to_get > len(dataset):
        raise ValueError(f'''Dataset is smaller than the amount of samples requested ({num_samples_to_get} samples). 
                         Maximum samples for this dataset is {len(dataset)}''')

    for batch_images, batch_masks in dataset:
        batch_size_1 = tf.shape(batch_images)[0].numpy()

        samples_needed_from_batch = min(batch_size_1, num_samples_to_get - samples_collected)

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

model = tf.keras.models.load_model(fine_tuned_model, 
                                   custom_objects={
                                    'dice_coef': dice_coef,
                                    'combined_loss': combined_loss,
                                    'dice_loss': dice_loss,
                                    'Patchify': Patchify,
                                    'PatchEncoder': PatchEncoder
                                    })

# Extract test dataset
dataset_path = save_datasets(new_directory_name_for_npy_files, None, None, None, None, None, None, return_folder_path=True)
x_test = np.load(os.path.join(dataset_path, 'test_images.npy'))
y_test = np.load(os.path.join(dataset_path, 'test_masks.npy'))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(parse_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

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