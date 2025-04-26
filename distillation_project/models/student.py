import tensorflow as tf
from tensorflow import keras
import keras_nlp
from config import BATCH_SIZE

def build_student(vectorizer, vocab_size, num_classes):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(), dtype=tf.string),
        vectorizer,
        keras.layers.Embedding(vocab_size, 128, mask_zero=True),
        keras_nlp.layers.TransformerEncoder(intermediate_dim=256, num_heads=4, dropout=0.1),
        keras_nlp.layers.TransformerEncoder(intermediate_dim=256, num_heads=4, dropout=0.1),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

get_student_model = lambda vectorizer, vocab_size, num_classes: build_student(vectorizer, vocab_size, num_classes)
