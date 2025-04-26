import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizerFast
from config import *

def load_and_prepare_data():
    df = pd.read_csv("./data/df_file.csv")
    df["text_length"] = df["Text"].apply(lambda x: len(str(x).split()))
    texts = df["Text"].values
    labels = df["Label"].values

    x_train, x_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    return x_train, x_val, y_train_enc, y_val_enc, label_encoder

def get_vectorizer(texts):
    vectorizer = keras.layers.TextVectorization(max_tokens=10000, output_mode='int', output_sequence_length=1024)
    vectorizer.adapt(texts)
    return vectorizer

def get_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")
