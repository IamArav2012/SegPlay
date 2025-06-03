import numpy as np
import os
os.environ['TF_DISABLE_MKL'] = '1'
import tensorflow as tf
import Pretrain_segmentator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split


old_model_path = 'pretrained_segmentator_weights.keras'
IMG_SIZE = 128
batch_size = 4
LEARNING_RATE = 1e-4
base_dir = r'D:\ML\Medical Datasets\Chest X-ray dataset for lung segmentation'

def load_and_preprocess():

    lung_images = []
    lung_masks = []

    for root_folder in os.listdir(base_dir):
        if root_folder == 'utis':
                continue
        
        root_path = os.path.join(base_dir, root_folder)
        if not os.path.isdir(root_path):
            continue

        img_dir = os.path.join(root_path, 'img')
        mask_dir = os.path.join(root_path, 'mask')  

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue

        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)

            if os.path.exists(mask_path):
            
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb') # Dataset is grayscale but pretrained model was on rgb images
                img = img_to_array(img) 
                
                mask = load_img(mask_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
                mask = img_to_array(mask)

                lung_images.append(img)
                lung_masks.append(mask)

                print(f'{len(lung_images)} sets of images and masks loaded')
            

    # Preprocess images and masks
    num_samples = len(lung_images)
    load_batch_size = 32

    for i in range(0, num_samples, load_batch_size):
        end = min(i + load_batch_size, num_samples)

        # Convert to array in batch
        images_batch = np.array(lung_images[i:end])
        masks_batch = np.array(lung_masks[i:end])

        # Process images
        images_batch = images_batch.astype(np.float32) / 255.0
        images_batch = images_batch.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

        # Process masks
        masks_batch = (masks_batch > 0.5).astype(np.float32)
        masks_batch = masks_batch.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        # Overwrite in original list/array
        lung_images[i:end] = images_batch
        lung_masks[i:end] = masks_batch

    
    x_train_temp, x_test, y_train_temp, y_test = train_test_split(lung_images, lung_masks, random_state=42, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.3, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(augment, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset


def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        mask = tf.image.rot90(mask, k)

    if tf.random.uniform(()) > 0.5:
        image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def combined_loss(y_true, y_pred):

    dice_loss = 1 - dice_coef(y_true, y_pred)
    bce_fn = tf.keras.losses.BinaryCrossentropy()
    bce_loss = bce_fn(y_true, y_pred)
    combo_loss = dice_loss * 0.65 + bce_loss * 0.35
    return combo_loss


if __name__ == '__main__':
    model = tf.keras.models.load_model(old_model_path)

    train_ds, val_ds, test_ds = load_and_preprocess()


    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=4, min_delta=0.01, restore_best_weights=True)

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics= [dice_coef],
    )

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stopping_cb])

    model.save('lung_mri_segmentator.keras')

    