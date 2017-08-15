import numpy as np
import tensorflow as tf

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, 1.0))

def f_props(layers, x):
    for layer in layers:
        x = layer.f_prop(x)
    return x

def build_vocab(file_path, target):
    vocab = set()
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()
        vocab.update(words)

    if target:
        w2i = {w: np.int32(i+2) for i, w in enumerate(vocab)}
        w2i['<s>'], w2i['</s>'] = np.int32(0), np.int32(1) # 文の先頭・終端記号
    else:
        w2i = {w: np.int32(i) for i, w in enumerate(vocab)}

    return w2i

def encode(sentence, w2i):
    encoded_sentence = []
    for w in sentence:
        encoded_sentence.append(w2i[w])
    return encoded_sentence

def load_data(file_path, vocab=None, w2i=None, target=True):
    if vocab is None and w2i is None:
        w2i = build_vocab(file_path, target)

    data = []
    for line in open(file_path, encoding='utf-8'):
        s = line.strip().split()
        if target:
            s = ['<s>'] + s + ['</s>']
        enc = encode(s, w2i)
        data.append(enc)
    i2w = {i: w for w, i in w2i.items()}
    return data, w2i, i2w

def load_toy_data(data_size):
    chars = list('abcdefghijklmnopqrstuvwxyz,. ')
    e_w2i = {}
    d_w2i = {'<s>': np.int32(0), '</s>': np.int32(1)}
    for i, char in enumerate(chars):
        e_w2i[char] = np.int32(i)
        d_w2i[char] = np.int32(i+2)
    e_i2w = {i: w for w, i in e_w2i.items()}
    d_i2w = {i: w for w, i in d_w2i.items()}

    sentence1 = 'i am so happy.'
    sentence2 = 'it is working.'
    instance_x1 = [e_w2i[char] for char in sentence1]
    instance_x2 = [e_w2i[char] for char in sentence2]
    train_X = [instance_x1, instance_x2]

    instance_y1 = [d_w2i[char] for char in ['<s>'] + list(sentence1) + ['</s>']]
    instance_y2 = [d_w2i[char] for char in ['<s>'] + list(sentence2) + ['</s>']]
    train_y = [instance_y1, instance_y2]

    return train_X, train_y, e_w2i, e_i2w, d_w2i, d_i2w
