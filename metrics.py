import tensorflow as tf
import keras

def iou_loss(y_true, y_pred):
    """
    Compute the Intersection over Union (IOU) loss between ground truth and predicted masks.
    
    Parameters:
        y_true (tf.Tensor): Ground truth labels or masks.
        y_pred (tf.Tensor): Predicted labels or masks.
    
    Returns:
        loss (tf.Tensor): IOU loss value.
    """
    # Reshape tensors to flatten them
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    # Compute IOU score
    iou = (intersection + 1.) / (union + 1.)
    
    # Compute IOU loss
    loss = 1 - iou
    
    return loss

def iou_bce_loss(y_true, y_pred):
    """
    Calculates the combined loss function of Intersection over Union (IoU) loss and Binary Crossentropy (BCE) loss.

    Parameters:
        y_true (tensor): The ground truth binary masks. Shape (batch_size, height, width, channels).
        y_pred (tensor): The predicted binary masks. Shape (batch_size, height, width, channels).

    Returns:
        tensor: The combined loss value.

    IoU Loss (Intersection over Union Loss):
    The IoU loss measures the similarity between two sets by calculating the size of their intersection divided by the size of their union.
    This loss function penalizes predictions that deviate from the ground truth in terms of how much they overlap.

    BCE Loss (Binary Crossentropy Loss):
    The BCE loss function measures the binary crossentropy between the ground truth and predicted values.
    This loss function penalizes discrepancies between predicted probabilities and true binary values.

    The combined loss value is calculated as the weighted sum of IoU loss and BCE loss, with both losses contributing equally.
    """
    # Calculate BCE loss using keras binary crossentropy
    bce_loss = keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Calculate IoU loss using custom function
    iou_loss_value = iou_loss(y_true, y_pred)

    # Combine both losses with equal weight
    combined_loss = 0.5 * bce_loss + 0.5 * iou_loss_value
    
    return combined_loss

def mean_iou(y_true, y_pred):
    """
    Calculates the mean Intersection over Union (IoU) metric for semantic segmentation tasks.

    Intersection over Union (IoU) is a common evaluation metric for segmentation tasks.
    It measures the overlap between the predicted segmentation mask and the ground truth mask.
    This function takes in the true segmentation masks (y_true) and the predicted masks (y_pred),
    calculates the IoU for each sample, and returns the mean IoU across all samples.

    Parameters:
    y_true (tensor): A tensor representing the ground truth segmentation masks.
                     It should have a shape of (batch_size, height, width, num_classes).
    y_pred (tensor): A tensor representing the predicted segmentation masks.
                     It should have the same shape as y_true.

    Returns:
    mean_iou (tensor): The mean Intersection over Union (IoU) across all samples in the batch.
                       It is a scalar tensor.
    """

    # Round the predicted masks to convert them into binary masks
    y_pred = tf.round(y_pred)
    
    # Calculate the intersection between the true and predicted masks
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    
    # Calculate the union of the true and predicted masks
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    
    # Add a smoothing term to avoid division by zero
    smooth = tf.ones(tf.shape(intersect))
    
    # Calculate IoU for each sample and take the mean
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))