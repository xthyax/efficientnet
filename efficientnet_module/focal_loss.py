import tensorflow as tf


def focal_loss(y_true, y_pred, class_weight=2, gamma=2.):
    # Took from: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
    """
    class_weight_tf = tf.constant(class_weight, dtype=tf.float32)

    epsilon = 1.e-9

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(class_weight_tf, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    loss = tf.reduce_mean(reduced_fl, axis=-1)
    return loss
