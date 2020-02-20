import tensorflow as tf

INF = -1e6


class TextCNN(tf.keras.models.Model):
    """Implement CNN for sentence classification
    https://arxiv.org/abs/1408.5882"""
    # def __init__(self, kernel_sizes, num_filters, embeddings=None, embedding_units=None,
    #              top_k_max_pooling=1, dropout=0.0):
    def __init__(self, config, pretrained_embeddings=None):
        super(TextCNN, self).__init__()
        self.config = config

        if pretrained_embeddings is None:
            self.embeddings = tf.keras.layers.Embedding(
                self.config.vocab_size, self.config.embedding_units
            )
        else:
            vocab_size, embedding_units = pretrained_embeddings.shape
            self.embeddings = tf.keras.layers.Embedding(
                vocab_size, embedding_units, weights=[pretrained_embeddings],
                trainable=False
            )

        self.convs = []
        for kernel_size in self.config.kernel_sizes:

            self.convs.append(
                tf.keras.layers.Conv1D(
                    filters=self.config.num_filters, kernel_size=kernel_size, activation="sigmoid"
                )
            )

        # self.pool = tf.keras.layers.MaxPool1D(self.config.pool_size)
        # Hidden size for text embedding
        # hidden_size = len(self.config.kernel_sizes) * self.config.num_filters * self.config.top_k_max_pooling

        # self.batch_norm = tf.keras.layers.BatchNormalization()

        # self.ffc1 = tf.keras.layers.Dense(self.config.num_filters)
        self.ffc = tf.keras.layers.Dense(self.config.num_classes)
        self.dropout = tf.keras.layers.Dropout(self.config.dropout)

    def call(self, inputs, training=False):
        """inputs shape: (batch_size, num_words)"""
        output_embeddings = self.embeddings(inputs)
        pooled_outputs = []
        for conv in self.convs:
            conv_output = conv(output_embeddings)
            pooled = tf.math.reduce_max(conv_output, axis=1)
            pooled_outputs.append(pooled)

        text_embeddings = tf.concat(pooled_outputs, axis=1)
        # text_embeddings = self.batch_norm(text_embeddings)
        logits = self.ffc(text_embeddings)
        if training:
            logits = self.dropout(logits, training=training)
        return logits
