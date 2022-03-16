from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow.keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder

from gensim.models import keyedvectors
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
import json
import requests, tqdm, zipfile
import joblib


MAX_LEN = 24

def main(args):
    # load miscs
    intent_test = pd.read_json(args.test_file)
    X_test = intent_test['text'].apply(lambda x: ' '.join(word_tokenize(x)))

    y_onehot = joblib.load('intent_onehot.joblib')

    tokenizer = joblib.load('intent_tokenizer.joblib')
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(
        X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post'
    )
    # load weights into model
    reconstructed_model = tf.keras.models.load_model('indent_model5.h5')

    # predict dataset
    y_predict = reconstructed_model.predict(X_test_padded)
    y_predict_onehot = tf.one_hot(tf.math.argmax(y_predict, axis=1), depth=150)
    res = y_onehot.inverse_transform(y_predict_onehot)

    # write prediction to file (args.pred_file)
    submission = pd.DataFrame({'id': intent_test['id'], 'intent': res.flatten()})
    submission.to_csv(args.pred_file, index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="path to the testing file.",
        default="./data/intent/test.json",
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="path to the output predictions.",
        default="./data/intent/pred.csv",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="path to the model file.",
        default="./intent_model5.h5",
    )

    # # data
    # parser.add_argument("--max_len", type=int, default=128)

    # # model
    # parser.add_argument("--hidden_size", type=int, default=512)
    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--dropout", type=float, default=0.1)
    # parser.add_argument("--bidirectional", type=bool, default=True)

    # # data loader
    # parser.add_argument("--batch_size", type=int, default=128)

    # parser.add_argument(
    #     "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    # )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
