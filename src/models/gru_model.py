from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Sequential


def build_gru(vocab_size, max_len, embed_dim=100, embedding_weights=None):
    """Bidirectional GRU classifier over a learned (or pretrained) embedding.

    embedding_weights: optional (vocab_size, embed_dim) array used to initialise
    the Embedding layer. The layer stays trainable either way.
    """
    embedding_kwargs = {}
    if embedding_weights is not None:
        embedding_kwargs["weights"] = [embedding_weights]
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=vocab_size, output_dim=embed_dim, **embedding_kwargs),
        Bidirectional(GRU(128, return_sequences=True)),
        Dropout(0.35),
        Bidirectional(GRU(64)),
        Dropout(0.25),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
