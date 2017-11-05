## Deep Recurrent Generative Decoder for Abstractive Text Summarization

Unofficial DyNet implementation of the paper Deep Recurrent Generative Decoder for Abstractive Text Summarization (EMNLP 2017)[1]

### 1. Requirements
- Python 3.6.0+
- DyNet 2.0+
- NumPy 1.12.1+
- scikit-learn 0.19.0+
- tqdm 4.15.0+

### 2. Prepare dataset
To get preprocedded gigaword corpus[2], run
```
sh download_gigaword_dataset.sh
```

### 3. Train
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `0`]
- `--n_epochs`: Number of epochs [default: `3`]
- `--n_train`: Number of training data (up to `3803957`) [default: `3803957`]
- `--n_valid`: Number of validation data (up to `189651`) [default: `189651`]
- `--vocab_size`: Vocabulary size [default: `60000`]
- `--batch_size`: Mini batch size [default: `32`]
- `--emb_dim`: Embedding size [default: `256`]
- `--hid_dim`: Hidden state size [default: `256`]
- `--lat_dim`: Latent state size [default: `256`]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `8192`]

#### Command example
```
python train.py --n_epochs 10
```

### 4. Test
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `0`]
- `--n_test`: Number of test data [default: `189651`]
- `--beam_size`: Beam size [default: `5`]
- `--max_len`: Maximum length of decoding [default: `100`]
- `--model_file`: Trained model file path [default: `./model_e1`]
- `--input_file`: Test file path [default: `./data/valid.article.filter.txt`]
- `--output_file`: Output file path [default: `./pred_y.txt`]
- `--w2i_file`: Word2Index file path [default: `./w2i.dump`]
- `--i2w_file`: Index2Word file path [default: `./i2w.dump`]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `1024`]

#### Command example
```
python test.py --beam_size 10
```

### 5. Evaluate
You can use pythonrouge[2] to measure the rouge scores.

### 6. Results
The model is trained with a full training data in [3].
ROUGE scores are obtained with `pythonrouge`.
#### 6.1. Gigaword (validation data)
|                 |ROUGE-1 (F1)|ROUGE-2 (F1)|ROUGE-L (F1)|
|-----------------|:-----:|:-----:|:-----:|
|My implementation| 43.27|19.17|40.47|

#### 6.2. DUC 2004
Work in progress.

#### 6.3. LCSTS
Work in progress.

### 7. Pretrained model
To get the pretrained model, run
```
sh download_gigaword_pretrained_model.sh
```
.

### Notes
- ROUGE scores are much higher than what the paper reported, but I don't know why. Please tell me if you know why!
- Original paper lacks some details and notations, and some points do not make sense, so this implementation may be different from the original one.
- Tensorflow implementation is in the directory `./tensorflow`, but not maintained.

### References
- [1] P. Li et al. 2017. Deep Recurrent Generative Decoder for Abstractive Text Summarization. In Proceedings of EMNLP 2017 \[[pdf\]](https://arxiv.org/abs/1708.00625)
- [2] pythonrouge: https://github.com/tagucci/pythonrouge
- [3] Gigaword/DUC2004 Corpus: https://github.com/harvardnlp/sent-summary
