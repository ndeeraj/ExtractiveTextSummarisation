import train_embeddings_bbc, preprocessing_bbc
import numpy as np
from nltk.tokenize import sent_tokenize

EMBED_SIZE = train_embeddings_bbc.EMBEDDINGS_SIZE
MAX_WORD_SIZE = preprocessing_bbc.MAX_THRESH


# mutates the provided numpy objects X, Y
def featurize_embed_from_df(df, X, Y, wvecs):
    for i in range(len(df)):
        inp_t = df.at[i, 'input']
        label = df.at[i, 'label']
        words = inp_t.split()
        wrd_count = len(words)
        for j in range(MAX_WORD_SIZE):
            if j >= wrd_count:
                break
            else:
                X[i, (j*EMBED_SIZE):(j*EMBED_SIZE)+EMBED_SIZE] = wvecs[words[j]]
        Y[i, 0] = label


# mutates the provided numpy objects X, Y
def featurize_X_from_text(text_X, wvecs):
    inf_X = np.zeros((1, MAX_WORD_SIZE * EMBED_SIZE))
    words = text_X.split()
    wrd_count = len(words)
    for j in range(MAX_WORD_SIZE):
        if j >= wrd_count:
            break
        else:
            inf_X[0, (j*EMBED_SIZE):(j*EMBED_SIZE)+EMBED_SIZE] = wvecs[words[j]]
    return inf_X


def create_inf_sents(inf_art_t):
    art_splits = inf_art_t.split('\n')
    orig_sent = []
    pre_sent = []
    WORD_COUNT = preprocessing_bbc.WORD_COUNT_THRES
    MAX_THRES = preprocessing_bbc.MAX_THRESH
    for split in art_splits:
        if len(split) == 0:
            continue
        art_sents = (sent_tokenize(split))
        for sent in art_sents:
            orig_sent.append(sent)
            inf_clean_t, inf_clean_count = preprocessing_bbc.apply_re(sent)
            if WORD_COUNT < inf_clean_count <= MAX_THRES:
                pre_sent.append(inf_clean_t.strip())

    return orig_sent, pre_sent