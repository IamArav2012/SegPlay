# lung_segmentator/pretraining.py

from os import environ
environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import tensorflow_datasets as tfds
from .config import IMG_SIZE, batch_size, dice_coef, dice_loss, pretrained_weights_path
from tensorflow.keras import layers, regularizers, models
from .layers import pass_through_vit


dropout_rate = 0.15
validation_split = 0.15
l2_reg_rate = 0.001
patch_size = 4
projection_dims = 256
learning_rate = 1e-3
epochs = 2

dataset, info = tfds.load("oxford_iiit_pet:4.0.0", with_info=True)

def preprocess(tensor):
    image = tf.image.resize(tensor['image'], (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.image.resize(tensor['segmentation_mask'], (IMG_SIZE, IMG_SIZE), method='nearest')
    mask = tf.cast(mask, tf.int32) - 1
    return image, mask

def augment(image, mask, hor_flip=True, contrast=True):
    if tf.random.uniform(()) > 0.5 and hor_flip:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5 and contrast:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, mask

def build_uvit():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    c1 = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg_rate))(inputs)
    b1 = layers.BatchNormalization()(c1)
    a1 = layers.Activation('relu')(b1)
    d1 = layers.Dropout(dropout_rate)(a1)
    p1 = layers.MaxPooling2D(2)(d1)

    c2 = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg_rate))(p1)
    b2 = layers.BatchNormalization()(c2)
    a2 = layers.Activation('relu')(b2)
    d2 = layers.Dropout(dropout_rate)(a2)
    p2 = layers.MaxPooling2D(2)(d2)

    c3 = layers.Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg_rate))(p2)
    b3 = layers.BatchNormalization()(c3)
    a3 = layers.Activation('relu')(b3)
    d3 = layers.Dropout(dropout_rate)(a3)
    p3 = layers.MaxPooling2D(2)(d3)

    c4 = layers.Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg_rate))(p3)
    b4 = layers.BatchNormalization()(c4)
    a4 = layers.Activation('relu')(b4)
    d4 = layers.Dropout(dropout_rate)(a4)

    vit_output = pass_through_vit(d4, patch_size=patch_size, input_size=[16, 16], ffn_hidden_units=[1024, projection_dims], projection_dims=projection_dims, transformer_blocks=4, attention_heads=6)
    vit_upsampled = layers.Conv2DTranspose(projection_dims, kernel_size=patch_size, strides=patch_size, padding='same')(vit_output)
    concat = layers.Concatenate()([vit_upsampled, d4])

    u1 = layers.Conv2DTranspose(512, (3,3), strides=(1,1), padding='same')(concat)
    b5 = layers.BatchNormalization()(u1)
    a5 = layers.Activation('relu')(b5)
    d5 = layers.Dropout(dropout_rate)(a5)

    u2 = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(d5)
    u2 = layers.Concatenate()([u2, d3])
    b6 = layers.BatchNormalization()(u2)
    a6 = layers.Activation('relu')(b6)
    d6 = layers.Dropout(dropout_rate)(a6)

    u3 = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(d6)
    u3 = layers.Concatenate()([u3, d2])
    b7 = layers.BatchNormalization()(u3)
    a7 = layers.Activation('relu')(b7)
    d7 = layers.Dropout(dropout_rate)(a7)

    u4 = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(d7)
    u4 = layers.Concatenate()([u4, d1])
    b8 = layers.BatchNormalization()(u4)
    a8 = layers.Activation('relu')(b8)
    d8 = layers.Dropout(dropout_rate)(a8)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d8)
    return models.Model(inputs, outputs)

if __name__ == '__main__':

    full_training_dataset = dataset['train']
    total_train_num_samples = info.splits['train'].num_examples
    full_training_dataset = full_training_dataset.shuffle(1000)
    train_dataset = full_training_dataset.skip(int(validation_split * total_train_num_samples))
    val_dataset = full_training_dataset.take(int(validation_split * total_train_num_samples))

    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_uvit()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(patience=1, min_delta=0.1, restore_best_weights=True)

    model.compile(
        optimizer=optimizer,
        loss=dice_loss,
        metrics=[dice_coef],
    )

    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[EarlyStopping])
    model.save(pretrained_weights_path)
    test_model_serialization = tf.keras.models.load_model(pretrained_weights_path)