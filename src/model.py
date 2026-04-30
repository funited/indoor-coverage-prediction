"""
Attention U-Net with a pretrained EfficientNet encoder.

The contracting path is a frozen-or-trainable EfficientNet (default B5).
The expanding path uses attention gates on the skip connections
(Oktay et al., 2018) and residual convolution blocks.

Output head is a 1x1 conv with linear activation, intended for regression
(e.g. predicting heatmap intensity). Swap to 'softmax' + categorical loss
for multi-class segmentation.
"""
from __future__ import annotations
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2


# All EfficientNet variants share the same skip-connection layer names,
# so the rest of the architecture is backbone-agnostic.
BACKBONES = {
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "EfficientNetB2": tf.keras.applications.EfficientNetB2,
    "EfficientNetB5": tf.keras.applications.EfficientNetB5,
    "EfficientNetB7": tf.keras.applications.EfficientNetB7,
}

SKIP_LAYERS = {
    "conv_128": "block2a_expand_activation",
    "conv_64":  "block3a_expand_activation",
    "conv_32":  "block4a_expand_activation",
    "conv_16":  "block5a_expand_activation",
    "bridge":   "top_activation",
}


# ---------- building blocks ----------

def res_conv_block(x, filter_size, size, dropout, batch_norm=False, reg=None):
    """Residual conv block: (Conv-BN-ReLU) x2 + dropout + 1x1 shortcut."""
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same",
                         kernel_regularizer=reg)(x)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same",
                         kernel_regularizer=reg)(conv)
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, (1, 1), padding="same",
                             kernel_regularizer=reg)(x)
    if batch_norm:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    return layers.Activation("relu")(layers.add([shortcut, conv]))


def gating_signal(input_tensor, out_size, batch_norm=False):
    """1x1 conv to align channels for the attention gate."""
    x = layers.Conv2D(out_size, (1, 1), padding="same")(input_tensor)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def attention_block(x, gating, inter_shape):
    """Additive attention gate (Oktay et al., 2018).

    Returns:
        (gated_features, attention_map): the second is kept for visualization;
        callers that don't need it can unpack as `result, _ = attention_block(...)`.
    """
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2),
                            padding="same")(x)
    shape_theta_x = K.int_shape(theta_x)

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding="same")(gating)
    upsample_g = layers.Conv2DTranspose(
        inter_shape, (3, 3),
        strides=(shape_theta_x[1] // shape_g[1],
                 shape_theta_x[2] // shape_g[2]),
        padding="same",
    )(phi_g)

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding="same")(act_xg)
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)

    upsample_psi = layers.UpSampling2D(
        size=(shape_x[1] // shape_sigmoid[1],
              shape_x[2] // shape_sigmoid[2]),
    )(sigmoid_xg)
    upsample_psi = layers.Lambda(
        lambda t: K.repeat_elements(t, shape_x[3], axis=3),
    )(upsample_psi)

    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1, 1), padding="same")(y)
    return layers.BatchNormalization()(result), sigmoid_xg


# ---------- model ----------

def build_attention_efficientunet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 1,
    backbone: str = "EfficientNetB5",
    dropout_rate: float = 0.5,
    batch_norm: bool = True,
    l2_reg: float = 0.01,
    encoder_trainable: bool = True,
    encoder_weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Attention U-Net with an EfficientNet encoder.

    Args:
        input_shape:  (H, W, C). H and W must be multiples of 32.
        num_classes:  output channels. 1 + 'linear' for regression (default).
        backbone:     one of BACKBONES keys.
        dropout_rate: dropout inside residual conv blocks.
        batch_norm:   enable BatchNorm in residual / gating blocks.
        l2_reg:       L2 regularization factor for decoder conv layers.
        encoder_trainable: freeze encoder when False (fast fine-tuning).
        encoder_weights:   "imagenet" or None.
    """
    if backbone not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Choose from: {list(BACKBONES)}"
        )

    reg = l2(l2_reg)
    FILTER_NUM, FILTER_SIZE, UP_SAMP_SIZE = 64, 3, 2

    inputs = layers.Input(input_shape, dtype=tf.float32)
    encoder = BACKBONES[backbone](
        include_top=False, weights=encoder_weights, input_tensor=inputs,
    )
    for layer in encoder.layers:
        layer.trainable = encoder_trainable

    skip_128 = encoder.get_layer(SKIP_LAYERS["conv_128"]).output
    skip_64  = encoder.get_layer(SKIP_LAYERS["conv_64"]).output
    skip_32  = encoder.get_layer(SKIP_LAYERS["conv_32"]).output
    skip_16  = encoder.get_layer(SKIP_LAYERS["conv_16"]).output
    bridge   = encoder.get_layer(SKIP_LAYERS["bridge"]).output

    def decoder_stage(prev, skip, filters):
        gating = gating_signal(prev, filters, batch_norm)
        att, _ = attention_block(skip, gating, filters)        # <-- bug fix
        up = layers.UpSampling2D(
            size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last",
        )(prev)
        up = layers.concatenate([up, att], axis=3)
        return res_conv_block(up, FILTER_SIZE, filters, dropout_rate,
                              batch_norm, reg=reg)

    x = decoder_stage(bridge, skip_16,  8 * FILTER_NUM)
    x = decoder_stage(x,      skip_32,  4 * FILTER_NUM)
    x = decoder_stage(x,      skip_64,  2 * FILTER_NUM)
    x = decoder_stage(x,      skip_128, 1 * FILTER_NUM)

    out = layers.Conv2D(num_classes, (1, 1), kernel_regularizer=reg)(x)
    out = layers.BatchNormalization(axis=3)(out)
    out = layers.Activation("linear")(out)

    return models.Model(inputs, out, name=f"Attention{backbone}UNet")


# ---------- compilation helper ----------

def compile_for_regression(
    model: tf.keras.Model,
    optimizer: str = "sgd",
    lr_start: float = 1e-2,
    lr_end: float = 1e-4,
    decay_steps: int = 1500,
    momentum: float = 0.9,
) -> tf.keras.Model:
    """Compile with polynomial-decay LR for MSE regression."""
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=lr_start,
        end_learning_rate=lr_end,
        decay_steps=decay_steps,
        power=1.0,
    )
    opts = {
        "sgd":     tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=lr_schedule),
        "adam":    tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    }
    if optimizer not in opts:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    model.compile(
        optimizer=opts[optimizer],
        loss="mean_squared_error",
        metrics=["mae"],   # accuracy is meaningless for regression; use MAE
    )
    return model


if __name__ == "__main__":
    m = build_attention_efficientunet(backbone="EfficientNetB5")
    m.summary()
