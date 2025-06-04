import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Lung_Segmentator import load_and_preprocess, combined_loss, dice_coef

model = tf.keras.models.load_model('lung_mri_segmentator.keras', 
                                   custom_objects={'combined_loss': combined_loss, 'dice_coef': dice_coef})

_, _, test_dataset = load_and_preprocess()

num_samples_to_get = 7
collected_images = []
collected_masks = []
samples_collected = 0

for batch_images, batch_masks in test_dataset:
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

y_pred = model.predict(final_images)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(final_images[0, ..., 0], cmap='gray')
plt.title('Lung Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(y_pred[0, ..., 0], cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(final_masks[0, ..., 0], cmap='gray')
plt.title('Actual Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

loss, dice = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}, Dice Coefficient: {dice:.4f}")

# Test Loss: 0.0981, Dice Coefficient: 0.9234 