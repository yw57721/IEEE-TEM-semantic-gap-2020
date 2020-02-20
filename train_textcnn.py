from model.textcnn import TextCNN
from config import Config
from absl import flags, app
import numpy as np
import tensorflow as tf
import os
import collections
from utils import get_logger
from dataset.amazon_review.glove_utils import load_vocab
from model.loss import SmoothedLoss
from metrics import MetricsAtTopK

logger = get_logger("train_textcnn")

FLAGS = flags.FLAGS

flags.DEFINE_string("config_file", "./conf/textcnn.yml", "config file")


def load_glove(config):
    glove_path = os.path.join(config.train_dir, "glove.npy")
    if not os.path.isfile(glove_path):
        logger.info("Glove vector not found. Please run dataset/amazon_review/data_generation.py")
        raise Exception()
    return np.load(glove_path)


def load_dataset(text_type, train_or_test, config):
    assert text_type in ["review", "need"]
    assert train_or_test in ["train", "test"]

    filepath = os.path.join(config.train_dir, "{}.{}.npz".format(text_type, train_or_test))
    data = np.load(filepath, allow_pickle=True)
    return data["text"], data["{}_label".format(config.attribute)]


def main(_):
    textcnn_config = Config(config_file=FLAGS.config_file)
    # textcnn_config.attribute = "screen"
    model_dir = os.path.join(textcnn_config.model_dir)
    result_dir = os.path.join(textcnn_config.result_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    # Load glove vectors
    glove_vectors = load_glove(textcnn_config)

    for attr in ["cpu", "ram", "screen", "hdd", "gpu"]:
        logger.info("Starting training for {}".format(attr))
        textcnn_config.attribute = attr
        checkpoint_dir = os.path.join(model_dir, attr)
        if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
        # Load data
        x_train, y_train = load_dataset("review", "train", textcnn_config)
        x_test, y_test = load_dataset("need", "test", textcnn_config)
        y_test = tf.cast(y_test, tf.int64)

        num_classes = max(set(y_train)) + 1
        textcnn_config.num_classes = num_classes

        # label_count = collections.Counter(y_train.tolist())
        # total_sample = y_train.size
        # class_weight = {}
        # for l, c in label_count.items():
        #     class_weight[l] = 1/c * total_sample / num_classes

        # Create model
        model = TextCNN(config=textcnn_config, pretrained_embeddings=glove_vectors)
        # Loss
        loss_fn = SmoothedLoss(smoothing_prob=textcnn_config.smoothing_prob)
        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(2e-3),
            loss=loss_fn
        )

        if textcnn_config.use_review:
            checkpoint_path = os.path.join(checkpoint_dir, attr+".ckpt")
            latest_ckp = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_ckp is None:
                callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                               save_weights_only=True)
                model.fit(
                    x_train, y_train,
                    # validation_data=(x_test, y_test),
                    epochs=textcnn_config.base_epochs, batch_size=128,
                    callbacks=[callbacks],
                    shuffle=True
                )
            else:
                model.load_weights(latest_ckp)

        ######## Fine tune
        x_train_2, y_train_2 = load_dataset("need", "train", textcnn_config)
        model.optimizer.learning_rate = 1e-3
        # num_classes = len(set(y_train_2))

        model.fit(
            x_train_2, y_train_2,
            # validation_data=(x_test, y_test),
            epochs=textcnn_config.finetune_epochs, batch_size=64,
            shuffle=True
        )

        # Prediction
        y_pred = model.predict(x_test)
        metrics = MetricsAtTopK(num_classes, y_test, y_pred)
        y_pred_np = np.asarray(y_pred, np.float32)
        np.savez(os.path.join(result_dir, attr), y_pred=y_pred_np,
                 precision=metrics.precision, recall=metrics.recall, f1=metrics.f1)
        logger.info("{} metrics: {}\n{}\n".format(attr, metrics.precision, metrics.recall))


if __name__ == '__main__':
    app.run(main)



