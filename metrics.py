import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession()


class MetricsAtTopK:
    def __init__(self, num_classes, y_true, y_pred):
        self.num_classes = num_classes
        # self.sess = tf.compat.v1.Session()
        # self.sess.run(tf.compat.v1.local_variables_initializer())
        self.precision = self.precision_at_k(y_true, y_pred)
        self.recall = self.recall_at_k(y_true, y_pred)
        self.f1 = (2 * self.precision * self.recall) / \
                        (self.precision + self.recall)
        # self.sess.close()

    def precision_at_k(self, y_true, y_pred):
        num_classes = self.num_classes
        precisions = []
        # with tf.compat.v1.Session() as sess:
        #     sess.run(tf.compat.v1.local_variables_initializer())
        for k in range(1, num_classes+1):
            _, pre = tf.compat.v1.metrics.precision_at_k(y_true, y_pred, k)
            sess.run(tf.compat.v1.local_variables_initializer())
            precisions.append(sess.run(pre))
        return np.asarray(precisions, np.float32)

    def recall_at_k(self, y_true, y_pred):
        num_classes = self.num_classes
        recalls = []
        # with tf.compat.v1.Session() as sess:
        #     sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        for k in range(1, num_classes + 1):
            _, rec = tf.compat.v1.metrics.recall_at_k(y_true, y_pred, k)
            sess.run(tf.compat.v1.local_variables_initializer())
            recalls.append(sess.run(rec))
        return np.asarray(recalls, np.float32)


if __name__ == '__main__':

    pass
