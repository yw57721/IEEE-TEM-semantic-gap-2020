import tensorflow as tf


class MaskedLoss(tf.keras.losses.Loss):

    def __init__(self):
        super(MaskedLoss, self).__init__()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none")

    def call(self, y_true, logits):
        loss = self.loss_fn(y_true, logits)
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return loss


class SmoothedLoss(tf.keras.losses.Loss):

    def __init__(self, smoothing_prob=0.1):
        super(SmoothedLoss, self).__init__()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.smoothing_prob = smoothing_prob

    def call(self, y_true, logits):
        # if self.smoothing_prob > 0:
        #     y_true = tf.cast(y_true, tf.float32)
        #     y_true *= (1 - self.smoothing_prob)
        #     y_true += (self.smoothing_prob / logits.shape[1])
        # print(y_true)
        loss = self.loss_fn(y_true, logits)

        return loss