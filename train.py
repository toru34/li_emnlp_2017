import os
import math
import time
import argparse
import pickle

import numpy as np
import _dynet as dy
from tqdm import tqdm
from sklearn.utils import shuffle

from utils import build_word2count, build_dataset
from layers import BiGRU, RecurrentGenerativeDecoder

RANDOM_STATE = 34
np.random.seed(RANDOM_STATE)

def main():
    parser = argparse.ArgumentParser(description='Deep Recurrent Generative Decoder for Abstractive Text Summarization in DyNet')

    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs [default: 3]')
    parser.add_argument('--n_train', type=int, default=3803957, help='Number of training examples (up to 3803957 in gigaword) [default: 3803957]')
    parser.add_argument('--n_valid', type=int, default=189651, help='Number of validation examples (up to 189651 in gigaword) [default: 189651])')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding size [default: 256]')
    parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
    parser.add_argument('--lat_dim', type=int, default=256, help='Latent size [default: 256]')
    parser.add_argument('--alloc_mem', type=int, default=8192, help='Amount of memory to allocate [mb] [default: 8192]')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    N_EPOCHS   = args.n_epochs
    N_TRAIN    = args.n_train
    N_VALID    = args.n_valid
    BATCH_SIZE = args.batch_size
    VOCAB_SIZE = 60000
    EMB_DIM    = args.emb_dim
    HID_DIM    = args.hid_dim
    LAT_DIM    = args.lat_dim
    ALLOC_MEM  = args.alloc_mem

    # File paths
    TRAIN_X_FILE = './data/train.article.txt'
    TRAIN_Y_FILE = './data/train.title.txt'
    VALID_X_FILE = './data/valid.article.filter.txt'
    VALID_Y_FILE = './data/valid.title.filter.txt'

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    dyparams.set_random_seed(RANDOM_STATE)
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Build dataset ====================================================================================
    w2c = build_word2count(TRAIN_X_FILE, n_data=N_TRAIN)
    w2c = build_word2count(TRAIN_Y_FILE, w2c=w2c, n_data=N_TRAIN)

    train_X, w2i, i2w = build_dataset(TRAIN_X_FILE, w2c=w2c, padid=False, eos=True, unksym='<unk>', target=False, n_data=N_TRAIN, vocab_size=VOCAB_SIZE)
    train_y, _, _     = build_dataset(TRAIN_Y_FILE, w2i=w2i, target=True, n_data=N_TRAIN)

    valid_X, _, _ = build_dataset(VALID_X_FILE, w2i=w2i, target=False, n_data=N_VALID)
    valid_y, _, _ = build_dataset(VALID_Y_FILE, w2i=w2i, target=True, n_data=N_VALID)

    VOCAB_SIZE = len(w2i)
    OUT_DIM = VOCAB_SIZE
    print(VOCAB_SIZE)

    # Build model ======================================================================================
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    V = model.add_lookup_parameters((VOCAB_SIZE, EMB_DIM))

    encoder = BiGRU(model, EMB_DIM, 2*HID_DIM)
    decoder = RecurrentGenerativeDecoder(model, EMB_DIM, 2*HID_DIM, LAT_DIM, OUT_DIM)

    # Train model =======================================================================================
    n_batches_train = math.ceil(len(train_X)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_X)/BATCH_SIZE)

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        # Train
        train_X, train_y = shuffle(train_X, train_y)
        loss_all_train = []
        for i in tqdm(range(n_batches_train)):
            # Create a new computation graph
            dy.renew_cg()
            encoder.associate_parameters()
            decoder.associate_parameters()

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            train_X_mb = train_X[start:end]
            train_y_mb = train_y[start:end]

            losses = []
            for x, t in zip(train_X_mb, train_y_mb):
                t_in, t_out = t[:-1], t[1:]

                # Encoder
                x_embs = [dy.lookup(V, x_t) for x_t in x]
                he = encoder(x_embs)

                # Decoder
                t_embs = [dy.lookup(V, t_t) for t_t in t_in]
                decoder.set_initial_states(he)
                y, KL = decoder(t_embs)

                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) + KL_t for y_t, t_t, KL_t in zip(y, t_out, KL)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward prop
            loss_all_train.append(mb_loss.value())

            # Backward prop
            mb_loss.backward()
            trainer.update()

        # Valid
        loss_all_valid = []
        for i in range(n_batches_valid):
            # Create a new computation graph
            dy.renew_cg()
            encoder.associate_parameters()
            decoder.associate_parameters()

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            valid_X_mb = valid_X[start:end]
            valid_y_mb = valid_y[start:end]

            losses = []
            for x, t in zip(valid_X_mb, valid_y_mb):
                t_in, t_out = t[:-1], t[1:]

                # Encoder
                x_embs = [dy.lookup(V, x_t) for x_t in x]
                he = encoder(x_embs)

                # Decoder
                t_embs = [dy.lookup(V, t_t) for t_t in t_in]
                decoder.set_initial_states(he)
                y, KL = decoder(t_embs)

                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) + KL_t for y_t, t_t, KL_t in zip(y, t_out, KL)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward prop
            loss_all_valid.append(mb_loss.value())

        print('EPOCH: %d, Train Loss: %.3f, Valid Loss: %.3f' % (
            epoch+1,
            np.mean(loss_all_train),
            np.mean(loss_all_valid)
        ))

        # Save model ======================================================================================
        dy.save('./model_e'+str(epoch+1), [V, encoder, decoder])
        with open('./w2i.dump', 'wb') as f_w2i, open('./i2w.dump', 'wb') as f_i2w:
            pickle.dump(w2i, f_w2i)
            pickle.dump(i2w, f_i2w)

if __name__ == '__main__':
    main()
