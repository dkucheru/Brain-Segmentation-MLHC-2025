import tensorflow as tf
from keras import Model, Input
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate
from keras.losses import Loss

class LossHUC(Loss):
    def __init__(self, parent_index, child_indexes, name="loss_hump_updated"):
        super().__init__(name=name)
        self.parent_index = parent_index
        self.child_indexes = child_indexes

    def call(self, y_true, y_pred):
        e = 1e-6

        # Part 1: for all annotated labels - calculate BCE
        y_true_is_observed = tf.cast(y_true != 2, tf.float32)  # 1's where y_true != 2, 0's elsewhere
        BCE_unmasked = y_true * tf.math.log(y_pred + e) + (1 - y_true) * tf.math.log(1 - y_pred + e)
        BCE = BCE_unmasked * y_true_is_observed

        # Part 2: for all unannotated labels (2s are only at sublabels indexes)
        # a) if parent label is 1
        # find which is the most probable child of the parent, assume true on it
        # For the rest of 2's assume negative
        # b) if parent label is 0
        # assume strong negative for all children

        # Gather the values from y_pred at child_indexes along the last dimension
        y_pred_child = tf.gather(y_pred, indices=self.child_indexes,
                                 axis=-1)  # Shape: (256, 256, 256, len(child_indexes))
        # Find the indexes of the maximum values within the selected child_indexes
        max_indices_within_child = tf.argmax(y_pred_child, axis=-1)  # Shape: (256, 256, 256)
        # Map these max indices back to the original last dimension indices
        max_indices_original = tf.gather(self.child_indexes, max_indices_within_child)  # Shape: (256, 256, 256)
        # Create a delta tensor with the same shape as y_pred, initialized with zeros
        delta = tf.zeros_like(y_pred)
        # Use one-hot encoding to set `1` at the positions of the maximum values within child indexes
        delta = tf.one_hot(max_indices_original, depth=y_pred.shape[-1], axis=-1)
        # Ensure the output shape is the same as y_pred
        delta = tf.cast(delta, y_pred.dtype)  # Shape: (256, 256, 256, 11)

        # Now, if observed parent label is 0 - the delta is all zeros, because we cannot assign soft positives to sub-labels
        parent_value = y_true[..., self.parent_index]  # Shape: (256, 256, 256)
        # Expand the dimensions to match the shape of y_true
        parent_tensor = tf.expand_dims(parent_value, axis=-1)  # Shape: (256, 256, 256, 1)
        # Tile the tensor along the last dimension to match y_true's shape
        filter_with_parent = tf.tile(parent_tensor, [1, 1, 1, y_true.shape[-1]])  # Shape: (256, 256, 256, 11)
        delta = delta * filter_with_parent

        # Part 3: calculate loss for the 'unobserved'
        hierarchical_unobserved = delta * tf.math.log(y_pred + e) + (1 - delta) * tf.math.log(1 - y_pred + e)
        y_true_is_unobserved = tf.cast(y_true == 2, tf.float32)  # 1's where y_true == 2, 0's elsewhere
        hierarchical_unobserved = y_true_is_unobserved * hierarchical_unobserved

        full_loss = BCE + hierarchical_unobserved
        full_loss = -full_loss
        averaged_full_loss = tf.math.reduce_mean(full_loss, axis=tuple(range(1, len(y_true.shape))))
        return averaged_full_loss


@tf.function
def weighted_normalized_loss(y_true, y_pred):
    e = 1e-6

    # Part 1: for all annotated labels - calculate BCE
    y_true_is_observed = tf.cast(y_true != 2, tf.float32)  # 1's where y_true != 2, 0's elsewhere
    BCE_unmasked = y_true * tf.math.log(y_pred + e) + (1 - y_true) * tf.math.log(1 - y_pred + e)
    BCE = BCE_unmasked * y_true_is_observed

    # Part 2: calculate loss for the 'unobserved'

    y_true_is_unobserved = tf.cast(y_true == 2, tf.float32)  # 1's where y_true == 2, 0's elsewhere

    # Extract and normalize values
    child_values = y_true_is_unobserved * y_pred

    sums = tf.reduce_sum(child_values, axis=-1, keepdims=True)
    normalized_weights = child_values / (sums + tf.keras.backend.epsilon())

    hierarchical_unobserved = normalized_weights * tf.math.log(y_pred + e) + (1 - normalized_weights) * tf.math.log(
        1 - y_pred + e)
    hierarchical_unobserved = y_true_is_unobserved * hierarchical_unobserved

    full_loss = BCE + hierarchical_unobserved
    full_loss = -full_loss
    averaged_full_loss = tf.math.reduce_mean(full_loss, axis=tuple(range(1, len(y_true.shape))))
    return averaged_full_loss


@tf.function
def weighted_loss(y_true, y_pred):
    e = 1e-6
    # Part 1: for all annotated labels - calculate BCE
    y_true_is_observed = tf.cast(y_true != 2, tf.float32)  # 1's where y_true != 2, 0's elsewhere
    BCE_unmasked = y_true * tf.math.log(y_pred + e) + (1 - y_true) * tf.math.log(1 - y_pred + e)
    BCE = BCE_unmasked * y_true_is_observed

    # Part 2: calculate loss for the 'unobserved'
    hierarchical_unobserved = y_pred * tf.math.log(y_pred + e) + (1 - y_pred) * tf.math.log(1 - y_pred + e)
    y_true_is_unobserved = tf.cast(y_true == 2, tf.float32)  # 1's where y_true == 2, 0's elsewhere
    hierarchical_unobserved = y_true_is_unobserved * hierarchical_unobserved

    full_loss = BCE + hierarchical_unobserved
    full_loss = -full_loss
    averaged_full_loss = tf.math.reduce_mean(full_loss, axis=tuple(range(1, len(y_true.shape))))
    return averaged_full_loss


@tf.function
def custom_loss_iu(y_true, y_pred):
    # 2 is the class label for "ignore" class
    # "do not ignore" labels are: 0, 1
    y_true_is_observed = tf.cast(y_true != 2, y_true.dtype)  # 1's where y_true != 2, 0's elsewhere
    e = 1e-6
    BCE_unmasked_unreduced = -(y_true * tf.math.log(y_pred + e) + (1 - y_true) * tf.math.log(1 - y_pred + e))
    BCE_masked_unreduced = BCE_unmasked_unreduced * y_true_is_observed
    res = tf.math.reduce_mean(BCE_masked_unreduced, axis=tuple(range(1, len(y_true.shape))))
    return res

def UNet2D(input_shape, normalizer=None, activation='sigmoid', num_classes=1):
    inputs = tf.keras.Input(input_shape)

    if normalizer is not None:
        inputs = normalizer(inputs)

    # min_shape = 32
    min_shape = 16  # BEST ONE
    # min_shape = 8

    conv1 = Conv2D(min_shape, (3, 3), padding='same')(inputs)

    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(min_shape, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(min_shape * 2, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(min_shape * 2, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(min_shape * 4, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(min_shape * 4, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(min_shape * 8, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(min_shape * 8, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(min_shape * 16, (3, 3), padding='same')(pool4)
    conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    conv5 = Conv2D(min_shape * 16, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Conv2DTranspose(min_shape * 8, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)

    conv6 = Conv2D(min_shape * 8, (3, 3), padding='same')(up6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    conv6 = Conv2D(min_shape * 8, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2DTranspose(min_shape * 4, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)

    conv7 = Conv2D(min_shape * 4, (3, 3), padding='same')(up7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(min_shape * 4, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2DTranspose(min_shape * 2, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)

    conv8 = Conv2D(min_shape * 2, (3, 3), padding='same')(up8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(min_shape * 2, (3, 3), padding='same')(conv8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = Conv2DTranspose(min_shape, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)

    conv9 = Conv2D(min_shape, (3, 3), padding='same')(up9)
    conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(min_shape, (3, 3), padding='same')(conv9)
    conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation=activation)(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
