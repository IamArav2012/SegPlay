# lung_segmentator/fine_tuning.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from config import pretrained_weights_path, batch_size, base_dir, dice_loss, dice_coef, combined_loss, save_datasets, new_directory_name_for_npy_files, fine_tuned_model, parse_image
from layers import Patchify, PatchEncoder
from sklearn.model_selection import train_test_split

learning_rate = 1e-4

def load_dataset():

    def augment_fn(image, mask):
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
    
    def make_dataset(image_paths, mask_paths, augment=False):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#--------------------------------------------------------------------------------------------------
    image_paths = []
    mask_paths = []

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
                image_paths.append(img_path)
                mask_paths.append(mask_path)

    x_train_temp, x_test, y_train_temp, y_test = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.3, random_state=42)

    save_datasets(new_directory_name_for_npy_files, x_train, x_val, x_test, y_train, y_val, y_test, return_folder_path=False)

    return make_dataset(x_train, y_train, augment=True), make_dataset(x_val, y_val), make_dataset(x_test, y_test)


if __name__ == '__main__':
    model = tf.keras.models.load_model(pretrained_weights_path,
    custom_objects={
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'Patchify': Patchify,
        'PatchEncoder': PatchEncoder
    })

    train_ds, val_ds, test_ds = load_dataset()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coef',  
        patience=6,
        min_delta=0.001,  
        mode='max',  
        restore_best_weights=True
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_dice_coef',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=fine_tuned_model,
        monitor="val_dice_coef",
        save_best_only=True,
        mode="max",
        )

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics= [dice_coef],
    )

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stopping_cb, lr_scheduler, checkpoint_cb])
    loss_value, dice_coef_value = model.evaluate(test_ds)
    print(f"Quick Evaluation: Loss: {loss_value:.4f}, Dice Coef: {dice_coef_value:.4f}")