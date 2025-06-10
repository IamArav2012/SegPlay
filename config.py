# config.py
import tensorflow as tf
import numpy as np
import os
import shutil

# Reusable constants / config
IMG_SIZE = 128
pretrained_weights_path = 'pretrained_segmentator_weights.keras'
fine_tuned_model = 'lung_mri_segmentator.keras'
batch_size = 4
base_dir =  r'D:\ML\Medical Datasets\Chest X-ray dataset for lung segmentation'

# Reusable vars / config
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

def save_datasets(new_folder_name, x_train, x_val, x_test, y_train, y_val, y_test, return_folder_path=False):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(this_dir, new_folder_name)
    if not return_folder_path:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Existing folder '{new_folder_name}' and all contents deleted.")
        
        os.mkdir(folder_path)
        print(f"New folder '{new_folder_name}' created at {folder_path}")
        
        # Save .npy files inside the new folder
        np.save(os.path.join(folder_path, 'train_images.npy'), x_train)
        np.save(os.path.join(folder_path, 'val_images.npy'), x_val)
        np.save(os.path.join(folder_path, 'test_images.npy'), x_test)
        np.save(os.path.join(folder_path, 'train_masks.npy'), y_train)
        np.save(os.path.join(folder_path, 'val_masks.npy'), y_val)
        np.save(os.path.join(folder_path, 'test_masks.npy'), y_test)
        
        print("All .npy files have been saved successfully.")

    else: 
        print('New folder path will be returned as requested')
        return folder_path   