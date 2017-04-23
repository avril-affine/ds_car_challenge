from keras import backend as K


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
