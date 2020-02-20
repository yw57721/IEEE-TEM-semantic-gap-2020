import numpy as np
from collections import OrderedDict, Counter
from utils import get_logger
import os
logger = get_logger("Glove")


def load_vocab(vocab_path):
    vocab = OrderedDict()
    with open(vocab_path) as fp:
        for i, line in enumerate(fp):
            w = line.strip()
            if w: vocab[w] = i
    return vocab


def load_and_save_glove(big_glove_path, original_vocab_path, train_dir=None):
    if not os.path.exists(train_dir): os.mkdir(train_dir)
    glove_vocab_path = os.path.join(train_dir, "vocab.txt")
    glove_vector_path = os.path.join(train_dir, "glove.npy")

    if os.path.isfile(glove_vocab_path) and os.path.isfile(glove_vector_path):
        logger.info("{} and {} already exist. loading..."
                    .format(glove_vocab_path, glove_vector_path))
        glove_vocab = load_vocab(glove_vocab_path)
        glove_vector = np.load(glove_vector_path)
        return glove_vocab, glove_vector

    logger.info("Loading original glove and extract part of it..")
    vocab = load_vocab(original_vocab_path)
    embeddings = OrderedDict()
    with open(big_glove_path, "r", encoding="utf-8") as fp:
        for line in fp:
            values = line.split(" ", 1)
            word = values[0]
            if word in vocab or word == "<p>":
                vector = values[1].split()
                vector = np.asarray(vector, dtype=np.float32)
                embeddings[word] = vector

    with open(glove_vocab_path, "w") as fp:
        fp.write("<pad>")
        fp.write("\n")
        fp.write("<unk>")
        fp.write("\n")
        for w in embeddings:
            fp.write(w)
            fp.write("\n")
    logger.info("Glove vocab saved at {}.".format(train_dir))

    padd_vec = embeddings["<p>"]
    vectors = [vec for vec in embeddings.values()]
    vectors = np.asarray(vectors, dtype=np.float32)
    unk_vector = np.mean(vectors, axis=0)

    padd_vec = np.reshape(padd_vec, (1, padd_vec.size))
    unk_vector = np.reshape(unk_vector, (1, unk_vector.size))
    vectors_arg = np.concatenate((padd_vec, unk_vector, vectors), axis=0)

    np.save(glove_vector_path, vectors_arg)
    logger.info("Small glove vector saved at {}.".format(glove_vector_path))

    return load_vocab(glove_vocab_path), vectors_arg


if __name__ == '__main__':
    import os
    basedir = os.path.dirname(os.path.abspath(__file__))
    train_dir = "/Users/linkai/data/hsu/amazon_review/train"

    load_and_save_glove(
        "/Users/linkai/data/glove/glove.840B.300d.txt",
        original_vocab_path="amazon_review/csv/vocab.txt",
        train_dir="/Users/linkai/data/hsu/amazon_review/train"
    )
    # print(len(word2index))
    # print(word2index)
    # print(len(embeddings))
    # print(padd_vec)
    # print(unk_vector)

    glove_path = os.path.join(train_dir, "glove.npy")
    glove_vector = np.load(glove_path)

    print(glove_vector[0])
    print(glove_vector[1])

