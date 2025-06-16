# lung_segmentator/layers.py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# -------------------
# Custom Layers
# -------------------

@register_keras_serializable()
class Patchify(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def build(self, input_shape):
        # Validate input shape (optional but helpful for debugging)
        if len(input_shape) != 4:
            raise ValueError(f"Expected input shape (batch, height, width, channels), got {input_shape}")
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        super().build(input_shape)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = tf.shape(patches)[-1]
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
        # input_shape: (batch_size, num_patches, patch_dim)
        self.projection.build(input_shape)
        self.pos_embedding.build((self.num_patches,))
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

class InputValidator:
    @staticmethod
    def validate_pass_through_vit(inputs, ffn_hidden_units, projection_dims, input_size):
        errors = []

        if not isinstance(ffn_hidden_units, list):
            errors.append("`ffn_hidden_units` must be a list where each element represents the units of a linear projection.")

        if isinstance(ffn_hidden_units, list) and ffn_hidden_units and ffn_hidden_units[-1] != projection_dims:
            errors.append("The last value in `ffn_hidden_units` must equal `projection_dims` for dimensional compatibility.")

        if len(inputs.shape) != 4:
            errors.append("`inputs` must be a 4D tensor: (batch_size, height, width, feature_maps).")

        if not isinstance(input_size, list):
            errors.append("`input_size` must be a list with two elements: [height, width].")

        if isinstance(input_size, list) and len(input_size) != 2:
            errors.append("`input_size` must contain exactly two elements: [height, width].")

        if errors:
            raise ValueError("ViT Input Validation Error(s):\n" + "\n".join(errors))
        
    # Add more Input warnings here for future blocks

# -------------------
# Functional Blocks
# -------------------

def mlp(x, hidden_units, dropout_rate=0.35):
    for units in hidden_units:
        x = layers.Dense(units=units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def pass_through_vit(inputs, patch_size, input_size, ffn_hidden_units, projection_dims, transformer_blocks, attention_heads):

    InputValidator.validate_pass_through_vit(inputs, ffn_hidden_units, projection_dims, input_size)
    patch_rows = input_size[0] // patch_size
    patch_cols = input_size[1] // patch_size
    num_patches = patch_rows * patch_cols    
    patches = Patchify(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dims)(patches)

    for _ in range(transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=projection_dims, dropout=0.1)(x1, x1)
        x2 = layers.Add()([encoded_patches, attention])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp_out = mlp(x3, ffn_hidden_units)
        encoded_patches = layers.Add()([x2, mlp_out])
    
    vit_output = layers.Reshape((patch_rows, patch_cols, projection_dims))(encoded_patches)

    return vit_output