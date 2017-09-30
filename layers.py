import _dynet as dy

from utils import dy_softplus, dy_log

class BiGRU:
    def __init__(self, model, emb_dim, hid_dim):
        pc = model.add_subcollection()

        # BiGRU
        self.BiGRUBuilder = dy.BiRNNBuilder(1, emb_dim, hid_dim, pc, dy.GRUBuilder)

        self.pc = pc
        self.spec = (emb_dim, hid_dim)

    def __call__(self, x):
        return self.BiGRUBuilder.transduce(x)

    def associate_parameters(self):
        pass

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim = spec
        return BiGRU(model, emb_dim, hid_dim)

    def param_collection(self):
        return self.pc

class RecurrentGenerativeDecoder:
    def __init__(self, model, emb_dim, hid_dim, lat_dim, out_dim):
        pc = model.add_subcollection()

        # First and Second GRUs
        self.firstGRUBuilder  = dy.GRUBuilder(1, emb_dim, hid_dim, pc)
        self.secondGRUBuilder = dy.GRUBuilder(1, emb_dim+hid_dim, hid_dim, pc)

        # Attention layer
        self._Wdhh = pc.add_parameters((hid_dim, hid_dim))
        self._Wehh = pc.add_parameters((hid_dim, hid_dim))
        self._ba   = pc.add_parameters((hid_dim), init=dy.ConstInitializer(0))
        self._v    = pc.add_parameters((hid_dim))

        # VAE
        # encoder
        self._Wezyh = pc.add_parameters((hid_dim, emb_dim))
        self._Wezzh = pc.add_parameters((hid_dim, lat_dim))
        self._Wezhh = pc.add_parameters((hid_dim, hid_dim))
        self._bezh  = pc.add_parameters((hid_dim), init=dy.ConstInitializer(0))
        # mean
        self._Wezhm = pc.add_parameters((lat_dim, hid_dim))
        self._bezm  = pc.add_parameters((lat_dim), init=dy.ConstInitializer(0))
        # var
        self._Whs  = pc.add_parameters((lat_dim, hid_dim))
        self._bezs = pc.add_parameters((lat_dim), init=dy.ConstInitializer(0))
        # decoder
        self._Wdyzh = pc.add_parameters((hid_dim, lat_dim))
        self._Wdzhh = pc.add_parameters((hid_dim, hid_dim))
        self._bdyh  = pc.add_parameters((hid_dim), init=dy.ConstInitializer(0))

        # Output layer
        self._Wdhy = pc.add_parameters((out_dim, hid_dim))
        self._bdhy = pc.add_parameters((out_dim), init=dy.ConstInitializer(0))

        # Initial state
        self._z_0 = pc.add_parameters((lat_dim))

        self.lat_dim = lat_dim
        self.pc = pc
        self.spec = (emb_dim, hid_dim, lat_dim, out_dim)

    def __call__(self, t, tm1s=None, test=False):
        if test:
            t_tm1   = t
            hd1_tm1 = tm1s[0]
            hd2_tm1 = tm1s[1]
            z_tm1   = tm1s[2]

            # First GRU
            hd1_t = self.firstGRUBuilder.initial_state().set_s([hd1_tm1]).add_input(t_tm1).output()

            # Attention layer
            e_t = dy.concatenate([dy.dot_product(self.v, dy.tanh(self.Wdhh*hd1_t + Wehh_he_j + self.ba)) for Wehh_he_j in self.Wehh_he])
            a_t = dy.softmax(e_t)
            c_t = dy.esum([dy.cmult(a_tj, he_j) for a_tj, he_j in zip(a_t, self.he)])

            # Second GRU
            hd2_t = self.secondGRUBuilder.initial_state().set_s([hd2_tm1]).add_input(dy.concatenate([c_t, t_tm1])).output()

            # VAE
            # encode
            hez_t  = dy.logistic(self.Wezyh*t_tm1 + self.Wezzh*z_tm1 + self.Wezhh*hd1_tm1 + self.bezh)
            mean_t = self.Wezhm*hez_t + self.bezm
            var_t  = dy_softplus(self.Whs*hez_t + self.bezs)

            eps = dy.random_normal(self.lat_dim)
            z_t = mean_t + dy.cmult(dy.sqrt(var_t), eps)

            # KL divergence
            KL_t = -0.5*dy.sum_elems(1 + dy_log(var_t) - dy.square(mean_t) - var_t)

            # decode
            hdy_t = dy.tanh(self.Wdyzh*z_t + self.Wdzhh*hd2_t + self.bdyh)

            # Output layer with softmax
            y_t = dy.softmax(self.Wdhy*hdy_t + self.bdhy)

            return y_t, hd1_t, hd2_t, z_t

        else:
            # First GRU
            hd1 = self.firstGRUBuilder.initial_state([self.hd1_0]).transduce(t)

            # Attention layer
            c = [] # context vectors
            for i, hd1_t in enumerate(hd1):
                e_t = dy.concatenate([dy.dot_product(self.v, dy.tanh(self.Wdhh*hd1_t + Wehh_he_j + self.ba)) for Wehh_he_j in self.Wehh_he])
                a_t = dy.softmax(e_t)
                c_t = dy.esum([dy.cmult(a_tj, he_j) for a_tj, he_j in zip(a_t, self.he)])
                c.append(c_t)

            # print(c)
            # Second GRU
            hd2_input = [dy.concatenate([c_t, t_tm1]) for c_t, t_tm1 in zip(c, t)]
            hd2 = self.secondGRUBuilder.initial_state([self.hd2_0]).transduce(hd2_input)

            # VAE & Output layer
            z_tm1 = self.z_0
            hd1_ = [self.hd1_0] + hd1[:-1] # [hd1_0, hd1_1, ..., hd1_Tm1]
            KL = []
            y = []
            for i, (t_tm1, hd1_tm1, hd2_t) in enumerate(zip(t, hd1_, hd2)):
                # VAE
                # encode
                hez_t = dy.logistic(self.Wezyh*t_tm1 + self.Wezzh*z_tm1 + self.Wezhh*hd1_tm1 + self.bezh)
                mean_t = self.Wezhm*hez_t + self.bezm
                var_t = dy_softplus(self.Whs*hez_t + self.bezs)

                eps = dy.random_normal(self.lat_dim)
                z_t = mean_t + dy.cmult(dy.sqrt(var_t), eps)
                z_tm1 = z_t

                # KL divergence
                KL_t = -0.5*dy.sum_elems(1 + dy_log(var_t) - dy.square(mean_t) - var_t)
                KL.append(KL_t)

                # decode
                hdy_t = dy.tanh(self.Wdyzh*z_t + self.Wdzhh*hd2_t + self.bdyh)

                # Output layer without softmax
                y_t = self.Wdhy*hdy_t + self.bdhy
                y.append(y_t)


            return y, KL

    def associate_parameters(self):
        self.Wdhh  = dy.parameter(self._Wdhh)
        self.Wehh  = dy.parameter(self._Wehh)
        self.ba    = dy.parameter(self._ba)
        self.v     = dy.parameter(self._v)
        self.Wezyh = dy.parameter(self._Wezyh)
        self.Wezzh = dy.parameter(self._Wezzh)
        self.Wezhh = dy.parameter(self._Wezhh)
        self.bezh  = dy.parameter(self._bezh)
        self.Wezhm = dy.parameter(self._Wezhm)
        self.bezm  = dy.parameter(self._bezm)
        self.Whs   = dy.parameter(self._Whs)
        self.bezs  = dy.parameter(self._bezs)
        self.Wdyzh = dy.parameter(self._Wdyzh)
        self.Wdzhh = dy.parameter(self._Wdzhh)
        self.bdyh  = dy.parameter(self._bdyh)
        self.Wdhy  = dy.parameter(self._Wdhy)
        self.bdhy  = dy.parameter(self._bdhy)
        self.z_0   = dy.parameter(self._z_0)

    def set_initial_states(self, he):
        hd_0 = dy.average(he)
        self.he = he
        self.hd1_0 = hd_0
        self.hd2_0 = hd_0
        self.Wehh_he = [self.Wehh*he_j for he_j in he]

    @staticmethod
    def from_spec(spec, model):
        emb_dim, hid_dim, lat_dim, out_dim = spec
        return RecurrentGenerativeDecoder(model, emb_dim, hid_dim, lat_dim, out_dim)

    def param_collection(self):
        return self.pc
