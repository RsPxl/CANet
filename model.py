from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def ca_block(input_feature, ratio=2, name="ca_block"):
    channel = input_feature._keras_shape[-1]
    h = input_feature._keras_shape[1]
    w = input_feature._keras_shape[2]
    print('1')
    x_h = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)
    x_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(x_h)
    x_w = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(input_feature)

    x_cat_conv_relu = Concatenate(axis=2)([x_w, x_h])
    x_cat_conv_relu = Conv2D(channel // ratio, kernel_size=1, strides=1, use_bias=False,
                             name="ca_block_conv1_" + str(name))(x_cat_conv_relu)
    x_cat_conv_relu = BatchNormalization(name="ca_block_bn_" + str(name))(x_cat_conv_relu)
    x_cat_conv_relu = Activation('relu')(x_cat_conv_relu)

    x_cat_conv_split_h, x_cat_conv_split_w = Lambda(lambda x: tf.split(x, num_or_size_splits=[h, w], axis=2))(
        x_cat_conv_relu)
    x_cat_conv_split_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(x_cat_conv_split_h)
    x_cat_conv_split_h = Conv2D(channel, kernel_size=1, strides=1, use_bias=False, name="ca_block_conv2_" + str(name))(
        x_cat_conv_split_h)
    x_cat_conv_split_h = Activation('sigmoid')(x_cat_conv_split_h)

    x_cat_conv_split_w = Conv2D(channel, kernel_size=1, strides=1, use_bias=False, name="ca_block_conv3_" + str(name))(
        x_cat_conv_split_w)
    x_cat_conv_split_w = Activation('sigmoid')(x_cat_conv_split_w)

    output = multiply([input_feature, x_cat_conv_split_h])
    output = multiply([output, x_cat_conv_split_w])
    return output


def unet_CA_GHM(pretrained_weights=None, input_size=(128, 128, 7)):
    inputs = Input(input_size)
    inputs_CA = ca_block(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs_CA)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation='softmax')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])  # sparse_categorical_crossentropy

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model