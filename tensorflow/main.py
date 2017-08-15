import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from utils import load_data, f_props, tf_log
from layers import Embedding, Dense3d, GRU, RVAE, Attention

random_state = 42

EMB_DIM = 31
HID_DIM = 32
LAT_DIM = 33
BATCH_SIZE = 2
N_EPOCHS = 2000
PADDING_ID = -1

# replace this with your train data
train_X, e_w2i, e_i2w = load_data('../data/train_input.txt', target=False)
train_y, d_w2i, d_i2w = load_data('../data/train_output.txt', target=True)
# replace this with your valid data
valid_X = train_X[:]
valid_y = train_y[:]

# Model ----------------------------------------------------------
e_vocab_size = len(e_w2i)
d_vocab_size = len(d_w2i)

x = tf.placeholder(tf.int32, [None, None], name='x')
m = tf.cast(tf.not_equal(x, -1), tf.float32)
t = tf.placeholder(tf.int32, [None, None], name='t')
t_in = t[:, :-1]
t_out = t[:, 1:]
t_out_one_hot = tf.one_hot(t_out, depth=d_vocab_size, dtype=tf.float32)

# Attention mask
ma = tf.where(
    condition=tf.equal(x, PADDING_ID),
    x=tf.ones_like(x, dtype=tf.float32)*np.float32(-1e+10),
    y=tf.ones_like(x, dtype=tf.float32)
)

encoder = [
    Embedding(e_vocab_size, EMB_DIM),
    GRU(EMB_DIM, HID_DIM, m),
    GRU(EMB_DIM, HID_DIM, m[:, ::-1])
]

x_emb = f_props(encoder[:1], x)
h_ef = f_props(encoder[1:2], x_emb)
h_eb = f_props(encoder[2:], x_emb[:, ::-1])[:, ::-1, :]
h_e = tf.concat([h_ef, h_eb], axis=2)
h_d1_0 = tf.reduce_mean(h_e, axis=1)
h_d2_0 = tf.reduce_mean(h_e, axis=1)

decoder = [
    Embedding(d_vocab_size, EMB_DIM),
    GRU(EMB_DIM, 2*HID_DIM, tf.ones_like(t_in, dtype='float32'), h_0=h_d1_0),
    Attention(2*HID_DIM, 2*HID_DIM, h_e, ma),
    GRU(EMB_DIM+2*HID_DIM, 2*HID_DIM, tf.ones_like(t_in, dtype='float32'), h_0=h_d2_0),
    RVAE(EMB_DIM, 2*HID_DIM, LAT_DIM),
    Dense3d(LAT_DIM+2*HID_DIM, HID_DIM, tf.nn.tanh),
    Dense3d(HID_DIM, d_vocab_size, tf.nn.softmax)
]

t_in_emb = f_props(decoder[:1], t_in)
h_d1 = f_props(decoder[1:2], t_in_emb)
h_d1__ = tf.concat([h_d1_0[:, None, :], h_d1], axis=1)[:, :-1, :]
c = f_props(decoder[2:3], h_d1)
h_d2 = f_props(decoder[3:4], tf.concat([t_in_emb, c], axis=2))
z, KL = f_props(decoder[4:5], [h_d1__, t_in_emb])
y = f_props(decoder[5:], tf.concat([z, h_d2], axis=2))

nll = -tf.reduce_mean(tf.reduce_sum(t_out_one_hot*tf_log(y), axis=[1, 2]))
kl = tf.reduce_mean(tf.reduce_sum(KL, axis=[1]))
cost = nll + kl

train = tf.train.AdamOptimizer().minimize(cost)

# Learn --------------------------------------------------------------------------
n_batches = len(train_X)//BATCH_SIZE

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(N_EPOCHS):
    # Train
    train_costs = []
    train_nlls = []
    train_kls = []
    for i in range(n_batches):
        start = i*BATCH_SIZE
        end = start + BATCH_SIZE

        train_X_mb = np.array(pad_sequences(train_X[start:end], padding='post', value=PADDING_ID))
        train_y_mb = np.array(pad_sequences(train_y[start:end], padding='post', value=PADDING_ID))

        _, train_cost, train_nll, train_kl = sess.run(
            [train, cost, nll, kl],
            feed_dict={
                x: train_X_mb,
                t: train_y_mb
            })
        train_costs.append(train_cost)
        train_nlls.append(train_nll)
        train_kls.append(train_kl)

    if (epoch+1)%100==0:
        # Valid
        valid_X_mb = np.array(pad_sequences(valid_X, padding='post', value=PADDING_ID))
        valid_y_mb = np.array(pad_sequences(valid_y, padding='post', value=PADDING_ID))

        valid_cost, valid_nll, valid_kl = sess.run([cost, nll, kl], feed_dict={x: valid_X_mb, t: valid_y_mb})
        print('EPOCH: %d, Train Cost: %.3f (NLL: %.3f, KL: %.3f), Valid Cost: %.3f (NLL: %.3f, KL: %.3f)'
             % (
                 epoch+1,
                 np.mean(train_costs),
                 np.mean(train_nlls),
                 np.mean(train_kls),
                 valid_cost,
                 valid_nll,
                 valid_kl
             )
             )


# Generate --------------------------------------------------------------------
t_0 = tf.constant(0)
h_d1_0_ph = tf.placeholder(tf.float32, [None, None], name='h_d1_0')
h_d2_0_ph = tf.placeholder(tf.float32, [None, None], name='h_d2_0')
z_0 = tf.zeros([BATCH_SIZE, LAT_DIM], tf.float32) # (batch_size, lat_dim)
y_0 = tf.placeholder(tf.int32, [None, None], name='y_0')
f_0 = tf.cast(tf.zeros_like(y_0[:, 0]), dtype=tf.bool)

f_0_size = tf.reduce_sum(tf.ones_like(f_0, dtype=tf.int32))
max_len = tf.placeholder(tf.int32, name='max_len')

def f_props_t(layers, x_t):
    for layer in layers:
        x_t = layer.f_prop_t(x_t)
    return x_t

def cond(t, h_d1_t, h_d2_t, z_t, y_t, f_t):
    num_true = tf.reduce_sum(tf.cast(f_t, tf.int32))
    unfinished = tf.not_equal(num_true, f_0_size)
    return tf.logical_and(t+1 < max_len, unfinished)

def body(t, h_d1_tm1, h_d2_tm1, z_tm1, y, f_tm1):
    y_tm1 = y[:, -1]

    decoder[1].h_0 = h_d1_tm1
    decoder[3].h_0 = h_d2_tm1
    decoder[4].z_0 = z_tm1
    y_emb_tm1 = f_props_t(decoder[:1], y_tm1)
    h_d1_t = f_props_t(decoder[1:2], y_emb_tm1)
    c_t = f_props_t(decoder[2:3], h_d1_t)
    h_d2_t = f_props_t(decoder[3:4], tf.concat([y_emb_tm1, c_t], axis=1))
    z_t = f_props_t(decoder[4:5], [h_d1_tm1, y_emb_tm1])
    y_t = f_props_t(decoder[5:], tf.concat([z_t, h_d2_t], axis=1))

    y_t_am = tf.cast(tf.argmax(y_t, axis=1), tf.int32)

    y = tf.concat([y, y_t_am[:, None]], axis=1)

    f_t = tf.logical_or(f_tm1, tf.equal(y_t_am, 1))

    return [t+1, h_d1_t, h_d2_t, z_t, y, f_t]

res = tf.while_loop(
    cond,
    body,
    loop_vars=[t_0, h_d1_0, h_d2_0, z_0, y_0, f_0],
    shape_invariants=[
        t_0.get_shape(), # t_0
        tf.TensorShape([None, None]), # h_d1_0
        tf.TensorShape([None, None]), # h_d2_0
        tf.TensorShape([None, None]), # z_0
        tf.TensorShape([None, None]), # y_0
        tf.TensorShape([None]) # f_0
    ]
)

valid_X_mb = pad_sequences(valid_X, padding='post', value=PADDING_ID)
y_0_ = np.zeros([BATCH_SIZE, 1], dtype='int32')
m_, h_e_, h_d1_0_, h_d2_0_ = sess.run([ma, h_e, h_d1_0, h_d2_0], feed_dict={x: valid_X_mb})

_, _, _, _, pred_ys, _ = sess.run(res, feed_dict={
    decoder[2].m: m_,
    decoder[2].h_e: h_e_,
    y_0: y_0_,
    h_d1_0_ph: h_d1_0_,
    h_d2_0_ph: h_d2_0_,
    max_len: 100
})

for true_x, true_y, pred_y in zip(valid_X, valid_y, pred_ys):
    print('true_x:', ' '.join([e_i2w[com] for com in true_x]))
    print('true_y:', ' '.join([d_i2w[com] for com in true_y[1:]]))
    print('pred_y:', ' '.join([d_i2w[com] for com in pred_y[1:]]))
    print()
