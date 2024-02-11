import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from PIL import Image 

from metrics import iou_bce_loss, mean_iou
from constants import SEED, BATCH_SIZE, VAL_BATCH_SIZE, IMAGE_SIZE, ADJUSTED_IMAGE_SIZE, TRAIN_DF_PATH
from constants import DETAIL_DF_PATH, TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, TRAIN_SAMPLES, VALID_SAMPLES, EPOCH_COUNT

INPUT_SHAPE = (ADJUSTED_IMAGE_SIZE, ADJUSTED_IMAGE_SIZE, 3)

def spatial_attention(input_tensor):
    """
    Implements spatial attention mechanism to enhance informative features.

    Args:
    input_tensor (tensor): Input tensor of shape (batch_size, height, width, channels).

    Returns:
    tensor: Scaled input tensor with spatial attention applied.
    """

    # Global Average Pooling to obtain channel-wise information
    squeeze = GlobalAveragePooling2D()(input_tensor)

    # Reshape to (batch_size, 1, 1, channels) to prepare for Dense layers
    squeeze = Reshape((1, 1, input_tensor.shape[-1]))(squeeze)

    # Apply two Dense layers to learn channel-wise dependencies
    excitation = Dense(input_tensor.shape[-1] // 2, activation='relu')(squeeze)
    excitation = Dense(input_tensor.shape[-1], activation='sigmoid')(excitation)

    # Reshape back to (batch_size, 1, 1, channels)
    excitation = Reshape((1, 1, input_tensor.shape[-1]))(excitation)

    # Element-wise multiplication with input tensor to apply attention
    scaled_input = Multiply()([input_tensor, excitation])

    return scaled_input


def conv_block(input, num_filters):
    """
    Creates a convolutional block consisting of two convolutional layers with batch normalization and ReLU activation.

    Parameters:
        input (tensor): Input tensor to the convolutional block.
        num_filters (int): Number of filters for the convolutional layers.

    Returns:
        tensor: Output tensor after passing through the convolutional block.

    """
    # First convolutional layer
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second convolutional layer
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    """
    Function to create a decoder block in a convolutional neural network.
    
    Parameters:
    input (Tensor): Input tensor to the decoder block.
    skip_features (Tensor): Skip connection tensor from the corresponding encoder block.
    num_filters (int): Number of filters for the convolutional layers in the block.
    
    Returns:
    Tensor: Output tensor from the decoder block.
    """
    # Upsampling through transpose convolution
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    
    # Concatenating skip connection features
    x = Concatenate()([x, skip_features])
    
    # Applying convolutional block
    x = conv_block(x, num_filters)
    
    return x

def model1():
    """
    Function to create a U-Net model based on pre-trained ResNet152 as encoder.

    Returns:
    tf.keras.Model: U-Net model
    """

    input_shape = INPUT_SHAPE

    def build_resnet152_unet(input_shape=input_shape, n=2, lr=0.0001):
        """
        Build the U-Net model with ResNet152 as encoder.

        Args:
        input_shape (tuple): Input shape of the model
        n (int): Number of output classes
        lr (float): Learning rate for optimizer

        Returns:
        tf.keras.Model: U-Net model
        """

        # Input
        inputs = Input(input_shape)

        # Pre-trained ResNet152 Model
        resnet152 = tf.keras.applications.ResNet152(include_top=False, weights="imagenet", input_tensor=inputs)

        resnet152.layers[0]._name = "input_1"
        resnet152._name = "Resnet152"

        # Encoder
        s1 = resnet152.get_layer("input_1").output
        s2 = resnet152.get_layer("conv1_relu").output
        s3 = resnet152.get_layer("conv2_block3_out").output
        s4 = resnet152.get_layer("conv3_block4_out").output

        # Bridge
        b1 = resnet152.get_layer("conv4_block6_out").output

        # Decoder
        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)

        # Output
        outputs = Conv2D(n, 1, padding="same", activation="softmax")(d4)

        model = Model(inputs, outputs, name="ResNet152_U-Net")

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=iou_bce_loss,
                      metrics=['accuracy', mean_iou])
        return model

    # Build and return the model
    model = build_resnet152_unet()
    return model

def model2():
    """
    This function constructs a U-Net model based on the DenseNet121 architecture.

    Returns:
    - tf.keras.Model: A compiled U-Net model.
    """

    input_shape = INPUT_SHAPE

    def build_densenet121_unet(input_shape=input_shape, n=2, lr=0.0001):
        """
        Builds the U-Net model architecture based on DenseNet121.

        Args:
        - input_shape (tuple): The shape of the input images.
        - n (int): Number of output classes.
        - lr (float): Learning rate for model training.

        Returns:
        - tf.keras.Model: Compiled U-Net model.
        """
        # Input layer
        inputs = Input(input_shape)

        # Pre-trained DenseNet121 Model
        densenet = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)

        # Naming layers
        densenet.layers[0]._name = "input_1"
        densenet._name = "Densenet121"

        # Encoder
        s1 = densenet.get_layer("input_1").output
        s2 = densenet.get_layer("conv1/relu").output
        s3 = densenet.get_layer("pool2_relu").output
        s4 = densenet.get_layer("pool3_relu").output

        # Bridge
        b1 = densenet.get_layer("pool4_relu").output

        # Decoder
        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)

        # Output
        outputs = Conv2D(n, 1, padding="same", activation="softmax")(d4)

        # Model compilation
        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=iou_bce_loss,
                      metrics=['accuracy', mean_iou])
        return model

    model = build_densenet121_unet()
    return model

def model3():
    """
    Description:
    This function builds a U-Net model with VGG19 encoder architecture for image segmentation.

    Returns:
    tf.keras.Model: The compiled U-Net model.

    Usage:
    model = model3()

    Example:
    model = model3()
    """

    input_shape = INPUT_SHAPE

    def build_vgg19_unet(input_shape=input_shape, n=2, lr=0.0001):
        """
        Description:
        Builds a U-Net model with VGG19 encoder architecture for image segmentation.
        This model incorporates spatial attention mechanisms to improve feature representation.

        Args:
        input_shape (tuple): The shape of the input image (height, width, channels).
        n (int): Number of output channels (classes).
        lr (float): Learning rate for the optimizer.

        Returns:
        tf.keras.Model: The compiled U-Net model.

        Usage:
        model = build_vgg19_unet(input_shape=(256, 256, 3), n=2, lr=0.0001)

        Example:
        model = build_vgg19_unet()
        """

        inputs = Input(input_shape)

        # Load VGG19 with pre-trained ImageNet weights
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=inputs)

        # Spatial Attention Mechanism
        # Enhance feature representation by focusing on relevant spatial locations
        s1 = spatial_attention(vgg19.get_layer('block1_conv2').output)
        s2 = spatial_attention(vgg19.get_layer('block2_conv2').output)
        s3 = spatial_attention(vgg19.get_layer('block3_conv3').output)
        s4 = spatial_attention(vgg19.get_layer('block4_conv3').output)
        b1 = spatial_attention(vgg19.get_layer('block5_conv3').output)

        # Decoder Block
        d1 = decoder_block(b1, s4, 512)
        d2 = decoder_block(d1, s3, 256)
        d3 = decoder_block(d2, s2, 128)
        d4 = decoder_block(d3, s1, 64)  # output layer

        # Final convolutional layer for segmentation output
        outputs = Conv2D(n, 1, padding='same', activation='sigmoid')(d4)

        # Build the model
        model = Model(inputs, outputs)

        # Compile the model with specified optimizer, loss function, and metrics
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=iou_bce_loss,
                      metrics=['accuracy', mean_iou])

        return model

    # Build and return the model
    model = build_vgg19_unet()
    return model


def model4():
    """
    Function to create a custom semantic segmentation model with ASPP (Atrous Spatial Pyramid Pooling) module.

    The model architecture is based on a combination of pre-trained ResNet152 and ASPP module
    for efficient semantic segmentation tasks.

    Returns:
        TensorFlow Keras model: Custom semantic segmentation model.
    """

    def ASPP(inputs):
        """
        Atrous Spatial Pyramid Pooling module.

        This module applies parallel dilated convolutions at multiple rates to capture
        multi-scale context information effectively.

        Args:
            inputs: Input tensor.

        Returns:
            TensorFlow tensor: Output tensor after ASPP module.
        """
        shape = inputs.shape

        # Global average pooling
        y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]), name='average_pooling')(inputs)
        y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
        y_pool = BatchNormalization(name=f'bn_1')(y_pool)
        y_pool = Activation('relu', name=f'relu_1')(y_pool)
        y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

        # Dilated convolutions
        y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(inputs)
        y_1 = BatchNormalization()(y_1)
        y_1 = Activation('relu')(y_1)

        y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(inputs)
        y_6 = BatchNormalization()(y_6)
        y_6 = Activation('relu')(y_6)

        y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(inputs)
        y_12 = BatchNormalization()(y_12)
        y_12 = Activation('relu')(y_12)

        y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(inputs)
        y_18 = BatchNormalization()(y_18)
        y_18 = Activation('relu')(y_18)

        # Concatenate features
        y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

        y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        return y

    def aspp_model(input_shape):
        """
        Build the custom semantic segmentation model.

        Args:
            input_shape (tuple): Input shape of the model.

        Returns:
            TensorFlow Keras model: Custom semantic segmentation model.
        """
        inputs = Input(input_shape)

        # Pre-trained ResNet50
        base_model = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_tensor=inputs)

        # Pre-trained ResNet50 Output
        image_features = base_model.get_layer('conv4_block6_out').output
        x_a = ASPP(image_features)
        x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

        # Get low-level features
        x_b = base_model.get_layer('conv2_block2_out').output
        x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
        x_b = BatchNormalization()(x_b)
        x_b = Activation('relu')(x_b)

        x = Concatenate()([x_a, x_b])

        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((4, 4), interpolation="bilinear")(x)

        # Outputs
        x = Conv2D(2, (1, 1), name='output_layer')(x)
        x = Activation('sigmoid')(x)

        # Model
        model = Model(inputs=inputs, outputs=x)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                      loss=iou_bce_loss,
                      metrics=['accuracy', mean_iou])
        return model

    input_shape = INPUT_SHAPE  # Assuming INPUT_SHAPE is defined somewhere else
    model = aspp_model(input_shape)

    return model