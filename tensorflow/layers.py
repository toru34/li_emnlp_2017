import numpy as np
import tensorflow as tf

from utils import tf_log

rng = np.random.RandomState(1234)

def get_W(size, name):
    return tf.Variable(
        rng.uniform(low=-0.08, high=0.08, size=size).astype('float32'),
        name=name
    )

def get_b(size, name):
    return tf.Variable(
        np.zeros(size).astype('float32'),
        name=name
    )

class Dense3d:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.W = get_W([in_dim, out_dim], 'W')
        self.b = get_b([out_dim], 'b')
        self.function = function

    def f_prop(self, x):
        return self.function(tf.einsum('ijk,kl->ijl', x, self.W) + self.b)

    def f_prop_t(self, x_t):
        return self.function(tf.matmul(x_t, self.W) + self.b)

class Embedding:
    def __init__(self, vocab_size, emb_dim):
        self.V = get_W([vocab_size, emb_dim], name='V')

    def f_prop(self, x):
        return tf.nn.embedding_lookup(self.V, x)

    def f_prop_t(self, x_t):
        return tf.nn.embedding_lookup(self.V, x_t)

class GRU:
    def __init__(self, in_dim, hid_dim, m, h_0=None):
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # Reset gate
        self.W_xr = get_W([in_dim, hid_dim], 'W_xr')
        self.W_hr = get_W([hid_dim, hid_dim], 'W_hr')
        self.b_r = get_b([hid_dim], 'b_r')

        # Update gate
        self.W_xz = get_W([in_dim, hid_dim], 'W_xz')
        self.W_hz = get_W([hid_dim, hid_dim], 'W_hz')
        self.b_z = get_b([hid_dim], 'b_z')

        self.W_xh = get_W([in_dim, hid_dim], 'W_xh')
        self.W_hh = get_W([hid_dim, hid_dim], 'W_hh')
        self.b_h = get_b([hid_dim], 'b_h')

        # Initial state
        self.h_0 = h_0

        # Mask
        self.m = m

    def f_prop(self, x):
        def fn(h_tm1, x_and_m):
            x_t = x_and_m[0]
            m_t = x_and_m[1]

            # Input gate
            r_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xr) + tf.matmul(h_tm1, self.W_hr) + self.b_r)

            # Update gate
            z_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xz) + tf.matmul(h_tm1, self.W_hz) + self.b_z)

            # Candidate activation
            g_t = tf.nn.tanh(tf.matmul(x_t, self.W_xh) + tf.matmul(r_t*h_tm1, self.W_hh) + self.b_h)

            # Activation
            h_t = z_t*h_tm1 + (1 - z_t)*g_t
            h_t = m_t[:, None]*h_t + (1 - m_t[:, None])*h_tm1 # Mask

            return h_t

        _x = tf.transpose(x, perm=[1, 0, 2])
        _m = tf.transpose(self.m)

        if self.h_0 == None:
            self.h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim]))

        h = tf.scan(fn=fn, elems=[_x, _m], initializer=self.h_0)
        return tf.transpose(h, perm=[1, 0, 2])

    def f_prop_t(self, x_t):
        # Input gate
        r_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xr) + tf.matmul(self.h_0, self.W_hr) + self.b_r)

        # Update gate
        z_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xz) + tf.matmul(self.h_0, self.W_hz) + self.b_z)

        # Candidate activation
        g_t = tf.nn.tanh(tf.matmul(x_t, self.W_xh) + tf.matmul(r_t*self.h_0, self.W_hh) + self.b_h)

        # Activation
        h_t = z_t*self.h_0 + (1 - z_t)*g_t

        return h_t

class RVAE:
    def __init__(self, emb_dim, hid_dim, lat_dim, z_0=None):
        self.W_zyh = get_W([emb_dim, hid_dim], 'W_zyh')
        self.W_zzh = get_W([lat_dim, hid_dim], 'W_zzh')
        self.W_zhh = get_W([hid_dim, hid_dim], 'W_zhh')
        self.b_zh = get_b([hid_dim], 'b_zh')

        self.W_zhm = get_W([hid_dim, lat_dim], 'W_zhm')
        self.W_hs = get_W([hid_dim, lat_dim], 'W_hs')
        self.b_zm = get_b([lat_dim], 'b_zm')
        self.b_zs = get_b([lat_dim], 'b_zs')

        self.emb_dim = emb_dim
        self.lat_dim = lat_dim

        self.z_0 = z_0

    def f_prop(self, x):
        def fn(tm1, inp):
            z_tm1 = tm1[0]
            KL_tm1 = tm1[1]
            h_d1_tm1 = inp[0]
            y_t = inp[1]

            h_z_t = tf.nn.sigmoid(
                tf.matmul(y_t, self.W_zyh)
                + tf.matmul(z_tm1, self.W_zzh)
                + tf.matmul(h_d1_tm1, self.W_zhh)
                + self.b_zh
            )

            mean_t = tf.matmul(h_z_t, self.W_zhm) + self.b_zm
            var_t = tf.nn.softplus(tf.matmul(h_z_t, self.W_hs) + self.b_zs)

            eps = tf.random_normal(shape=tf.shape(mean_t), dtype=tf.float32)
            KL_t = -0.5*tf.reduce_sum(1 + tf_log(var_t) - mean_t**2 - var_t, axis=1)
            z_t = mean_t + tf.sqrt(var_t)*eps

            return [z_t, KL_t]

        h_d1 = x[0]
        y = x[1]
        h_d1_ = tf.transpose(h_d1, perm=[1, 0, 2])
        y_ = tf.transpose(y, perm=[1, 0, 2])

        if self.z_0 == None:
            z_0 = tf.matmul(y[:, 0, :], tf.zeros([self.emb_dim, self.lat_dim]))
        KL_0 = tf.matmul(y[:, 0, :], tf.zeros([self.emb_dim, self.lat_dim]))[:, 0]

        z, KL = tf.scan(fn=fn, elems=[h_d1_, y_], initializer=[z_0, KL_0])

        return tf.transpose(z, perm=[1, 0, 2]), tf.transpose(KL)

    def f_prop_t(self, x_t):
        h_d1_tm1 = x_t[0]
        y_t = x_t[1]

        h_z_t = tf.nn.sigmoid(
            tf.matmul(y_t, self.W_zyh)
            + tf.matmul(self.z_0, self.W_zzh)
            + tf.matmul(h_d1_tm1, self.W_zhh)
            + self.b_zh
        )

        mean_t = tf.matmul(h_z_t, self.W_zhm) + self.b_zm
        var_t = tf.nn.softplus(tf.matmul(h_z_t, self.W_hs) + self.b_zs)

        eps = tf.random_normal(shape=tf.shape(mean_t), dtype=tf.float32)
        z_t = mean_t + tf.sqrt(var_t)*eps

        return z_t

class Attention:
    def __init__(self, ehid_dim, dhid_dim, h_e, m):
        self.W_dhh = get_W([dhid_dim, dhid_dim], 'W_dhh')
        self.W_ehh = get_W([ehid_dim, dhid_dim], 'W_ehh')
        self.v = tf.Variable(tf.random_uniform([dhid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='v')
        self.b_a = get_b([dhid_dim], 'b_a')

        self.h_e = h_e # Encoder hidden state
        self.m = m     # Encoder mask: (batch_size, enc_length)

    def f_prop(self, h_d):
        def fn(_, h_d_t):
            tmp_t = tf.nn.tanh(h_d_t[:, None, :] + self.h_e_ + self.b_a)

            score_t = tf.einsum('ijk,k->ij', tmp_t, self.v)
            score_t = score_t*self.m # Mask

            weight_t = tf.nn.softmax(score_t)

            return weight_t

        h_d_ = tf.einsum('ijk,kl->jil', h_d, self.W_dhh)
        self.h_e_ = tf.einsum('ijk,kl->ijl', self.h_e, self.W_ehh)

        weight_0 = tf.zeros_like(self.h_e[:, :, 0])

        weight = tf.scan(fn=fn, elems=h_d_, initializer=weight_0)

        # Context vector
        c = tf.einsum('ijk,jkl->jil', weight, self.h_e)
        return c

    def f_prop_t(self, h_d1_t):
        h_d1_t = tf.matmul(h_d1_t, self.W_dhh)
        self.h_e_ = tf.einsum('ijk,kl->ijl', self.h_e, self.W_ehh)
        tmp_t = tf.nn.tanh(h_d1_t[:, None, :] + self.h_e_ + self.b_a)

        score_t = tf.einsum('ijk,k->ij', tmp_t, self.v)
        score_t = score_t*self.m

        weight_t = tf.nn.softmax(score_t)

        # Context vector
        c_t = tf.einsum('ij,ijk->ik', weight_t, self.h_e)

        return c_t
