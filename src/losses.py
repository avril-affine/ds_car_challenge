from keras import backend as K


def weighted_logloss(y_true, y_pred, zero_weight):
    y_true_ = K.flatten(y_true)
    y_pred_ = K.clip(K.flatten(y_pred), K.epsilon(), 1 - K.epsilon())

    zero_mask = K.cast(K.equal(y_true_, 0.), 'float32') * zero_weight
    one_mask = K.cast(K.equal(y_true_, 1.), 'float32')
    weights = zero_mask + one_mask

    return -K.mean(y_true_ * K.log(y_pred_) + weights * (1 - y_true_) * K.log(1 - y_pred_))


def jaccard(y_true, y_pred):
    y_true_ = K.flatten(y_true)
    y_pred_ = K.flatten(y_pred)
    intersection = K.sum(y_true_ * y_pred_)
    return (intersection + 1.) / (K.sum(y_true_) + K.sum(y_pred_) - intersection + 1.0)


def dice_coef(y_true, y_pred):
    y_true_ = K.flatten(y_true)
    y_pred_ = K.flatten(y_pred)
    intersection = K.sum(y_true_ * y_pred_)
    return (2. * intersection + 1.) / (K.sum(y_true_) + K.sum(y_pred_) + 1.0)
