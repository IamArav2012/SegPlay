import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.saving import register_keras_serializable

REGULARIZER_RATE = 0.001 
DROPOUT_RATE = 0.15
VALIDATION_SPLIT = 0.15
batch_size = 4
IMG_SIZE = 128
L2_REG_RATE = 0.001
vit_mlp_units = [512, 256]
patch_size = 4
projection_dims = 256
transformer_blocks = 4
attention_heads = 6
LEARNING_RATE = 1e-3
EPOCHS = 2

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

@register_keras_serializable()
class Patchify(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size
        })
        return config


@register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dims, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dims = projection_dims
        self.projection = layers.Dense(units=projection_dims)
        self.pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dims)

    def build(self, input_shape):
        self.projection.build(input_shape)
        self.pos_embedding.build((None,))
        super().build(input_shape)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.pos_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dims': self.projection_dims
        })
        return config

def mlp(x, hidden_units, dropout_rate=0.35):
    for units in hidden_units:
        x = layers.Dense(units=units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def pass_through_vit(inputs):
    num_patches = (16 // patch_size) ** 2
    patches = Patchify(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dims)(patches)
    for _ in range(transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=projection_dims, dropout=0.1)(x1, x1)
        x2 = layers.Add()([encoded_patches, attention])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp_out = mlp(x3, [4 * projection_dims, projection_dims])
        encoded_patches = layers.Add()([x2, mlp_out])
    vit_output = layers.Reshape((16 // patch_size, 16 // patch_size, projection_dims))(encoded_patches)
    return vit_output

def build_uvit():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    c1 = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG_RATE))(inputs)
    b1 = layers.BatchNormalization()(c1)
    a1 = layers.Activation('relu')(b1)
    d1 = layers.Dropout(DROPOUT_RATE)(a1)
    p1 = layers.MaxPooling2D(2)(d1)

    c2 = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG_RATE))(p1)
    b2 = layers.BatchNormalization()(c2)
    a2 = layers.Activation('relu')(b2)
    d2 = layers.Dropout(DROPOUT_RATE)(a2)
    p2 = layers.MaxPooling2D(2)(d2)

    c3 = layers.Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG_RATE))(p2)
    b3 = layers.BatchNormalization()(c3)
    a3 = layers.Activation('relu')(b3)
    d3 = layers.Dropout(DROPOUT_RATE)(a3)
    p3 = layers.MaxPooling2D(2)(d3)

    c4 = layers.Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG_RATE))(p3)
    b4 = layers.BatchNormalization()(c4)
    a4 = layers.Activation('relu')(b4)
    d4 = layers.Dropout(DROPOUT_RATE)(a4)

    vit_output = pass_through_vit(d4)
    vit_upsampled = layers.Conv2DTranspose(projection_dims, kernel_size=patch_size, strides=patch_size, padding='same')(vit_output)
    concat = layers.Concatenate()([vit_upsampled, d4])

    u1 = layers.Conv2DTranspose(512, (3,3), strides=(1,1), padding='same')(concat)
    b5 = layers.BatchNormalization()(u1)
    a5 = layers.Activation('relu')(b5)
    d5 = layers.Dropout(DROPOUT_RATE)(a5)

    u2 = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(d5)
    u2 = layers.Concatenate()([u2, d3])
    b6 = layers.BatchNormalization()(u2)
    a6 = layers.Activation('relu')(b6)
    d6 = layers.Dropout(DROPOUT_RATE)(a6)

    u3 = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(d6)
    u3 = layers.Concatenate()([u3, d2])
    b7 = layers.BatchNormalization()(u3)
    a7 = layers.Activation('relu')(b7)
    d7 = layers.Dropout(DROPOUT_RATE)(a7)

    u4 = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(d7)
    u4 = layers.Concatenate()([u4, d1])
    b8 = layers.BatchNormalization()(u4)
    a8 = layers.Activation('relu')(b8)
    d8 = layers.Dropout(DROPOUT_RATE)(a8)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d8)
    return models.Model(inputs, outputs)

@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

if __name__ == '__main__':

    full_training_dataset = dataset['train']
    total_train_num_samples = info.splits['train'].num_examples
    full_training_dataset = full_training_dataset.shuffle(1000)
    train_dataset = full_training_dataset.skip(int(VALIDATION_SPLIT * total_train_num_samples))
    val_dataset = full_training_dataset.take(int(VALIDATION_SPLIT * total_train_num_samples))
    test_dataset = dataset['test']

    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_uvit()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(patience=1, min_delta=0.1, restore_best_weights=True)

    model.compile(
        optimizer=optimizer,
        loss=dice_loss,
        metrics=[dice_coef],
    )

    model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[EarlyStopping])
    model.save('pretrained_segmentator_weights.keras')
    test_model_serialization = tf.keras.models.load_model('pretrained_segmentator_weights.keras')

    # 782/782 ━━━━━━━━━━━━━━━━━━━━ 747s 953ms/step - dice_coef: 0.8981 - loss: 0.1028 - val_dice_coef: 0.8925 - val_loss: 0.1076