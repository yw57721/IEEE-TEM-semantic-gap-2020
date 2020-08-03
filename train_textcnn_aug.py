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

tf.random.set_random_seed(123456)

logger = get_logger("train_textcnn")

FLAGS = flags.FLAGS

flags.DEFINE_string("config_file", "./conf/textcnn_aug.yml", "config file")


def load_glove(config):
    glove_path = os.path.join(config.train_dir, "glove.npy")
    if not os.path.isfile(glove_path):
        logger.info("Glove vector {} not found. Please run dataset/amazon_review/data_generation.py".format(glove_path))
        raise Exception()
    return np.load(glove_path)


def load_dataset(text_type, train_or_test, config, attr):
    assert text_type in ["base", "need"]
    assert train_or_test in ["train", "test"]

    if text_type == "base":
        filepath = os.path.join(config.train_dir, "base.{}.train.npz".format(attr))
    else:
        filepath = os.path.join(config.train_dir, "need.{}.npz".format(train_or_test))

    # filepath = os.path.join(config.train_dir, "{}.{}.npz".format(text_type, train_or_test))
    # if not os.path.isfile(filepath):
    #     # assert attr is not None
    #     filepath = os.path.join(config.train_dir, "{}.{}.npz".format(attr, train_or_test))
    #
    # if text_type == "review" and train_or_test == "train":
    #     filepath = os.path.join(config.train_dir, "review.{}.{}.npz".format(attr, train_or_test))

    data = np.load(filepath, allow_pickle=True)

    label_name = "{}_label".format(attr)
    if label_name in data:
        return data["text"], data[label_name]
    return data["text"], data["label"]


def run_experiment(run_id=None):
    textcnn_config = Config(config_file=FLAGS.config_file)
    # textcnn_config.attribute = "screen"
    model_dir = os.path.join(textcnn_config.model_dir)
    result_dir = os.path.join(textcnn_config.result_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    # Load glove vectors
    pretrained_vectors = None
    if textcnn_config.use_glove:
        pretrained_vectors = load_glove(textcnn_config)
    # Load vocab
    vocab_file = os.path.join(textcnn_config.train_dir, "vocab.txt")
    vocab = load_vocab(vocab_file)
    textcnn_config.vocab_size = len(vocab)

    attribute = textcnn_config.attribute
    if attribute not in ["cpu", "ram", "screen", "hdd", "gpu"]:
        attribute = ["cpu", "ram", "screen", "hdd", "gpu"]
    else:
        # Make it a list.
        attribute = [attribute]

    # for attr in ["cpu", "ram", "screen", "hdd", "gpu"]:
    for attr in attribute:
        logger.info("Starting training for {}".format(attr))
        textcnn_config.attribute = attr
        checkpoint_dir = os.path.join(model_dir, attr)
        if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
        # Load data
        x_train, y_train = load_dataset("base", "train", textcnn_config, attr)
        x_test, y_test = load_dataset("need", "test", textcnn_config, attr)

        num_classes = max(set(y_train)) + 1
        textcnn_config.num_classes = num_classes

        # label_count = collections.Counter(y_train.tolist())
        # total_sample = y_train.size
        # class_weight = {}
        # for l, c in label_count.items():
        #     class_weight[l] = 1/c * total_sample / num_classes

        # Create model
        # model = TextCNN(config=textcnn_config, pretrained_embeddings=glove_vectors)
        model = TextCNN(config=textcnn_config, pretrained_embeddings=pretrained_vectors)
        # Loss
        loss_fn = SmoothedLoss(smoothing_prob=textcnn_config.smoothing_prob)
        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        lr = textcnn_config.learning_rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            # optimizer=tf.keras.optimizers.SGD(lr),
            loss=loss_fn
        )

        if textcnn_config.use_review:
            checkpoint_path = os.path.join(checkpoint_dir, attr+".ckpt")
            latest_ckp = tf.train.latest_checkpoint(checkpoint_dir)
            # if latest_ckp is None:
            if True:
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
        x_train_2, y_train_2 = load_dataset("need", "train", textcnn_config, attr)
        # lr *= 0.5
        # model.optimizer.learning_rate = lr
        # num_classes = len(set(y_train_2))

        checkpoint_path = os.path.join(checkpoint_dir, attr + ".final.ckpt")
        callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                       save_weights_only=True)
        # Freeze all layers except the dense layer
        for layer in model.layers:
            if layer.name != "dense":
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            # optimizer=tf.keras.optimizers.SGD(lr),
            loss=loss_fn
        )

        model.fit(
            x_train_2, y_train_2,
            # validation_data=(x_test, y_test),
            epochs=textcnn_config.finetune_epochs, batch_size=64,
            callbacks=[callbacks],
            shuffle=True
        )

        # Prediction
        y_test = tf.cast(y_test, tf.int64)
        y_pred = model.predict(x_test)
        metrics = MetricsAtTopK(num_classes, y_test, y_pred)
        y_pred_np = np.asarray(y_pred, np.float32)

        result_file = os.path.join(result_dir, attr)
        if run_id is not None:
            if isinstance(run_id, int):
                run_id = str(run_id)
            if len(run_id) < 2:
                run_id = "0" + run_id
            result_file = os.path.join(result_dir, attr+"_"+run_id)

        np.savez(result_file, y_pred=y_pred_np,
                 precision=metrics.precision, recall=metrics.recall, f1=metrics.f1)
        logger.info("{} metrics: {}\n{}\n".format(attr, metrics.precision, metrics.recall))


def main(_):
    # run_multiple = True
    # if run_multiple:
    #     num_run = 100
    #     for i in range(num_run):
    #         run_experiment(i)
    # else:
    #     run_experiment()
    run_experiment()


if __name__ == '__main__':
    app.run(main)



