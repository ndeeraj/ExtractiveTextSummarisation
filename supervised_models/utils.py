import train_embeddings_bbc
import preprocessing_bbc
import numpy as np
from nltk.tokenize import sent_tokenize

"""
This module contains common utilities used by all the supervised models to generate the 
input features (X)
"""

EMBED_SIZE = train_embeddings_bbc.EMBEDDINGS_SIZE
MAX_WORD_SIZE = preprocessing_bbc.MAX_THRESH


def featurize_embed_from_df(df, X, Y, wvecs):
    """
    For the data in provided dataframe [sentences, labels], fills the feature vectors in X
    (based on the embeddings provided by wvecs) and their corresponding labels in Y.

    Parameters:
        df (pandas dataframe): each row is expected to have sentence text, label
        X (numpy array): shape should be (len(df), MAX_WORD_SIZE*EMBED_SIZE)
        Y (numpy array): shape should be (len(df), 1)
        wvecs (dict): dict of embeddings, but ideally should be KeyedVectors from gensim
    """

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


# same purpose as featurize_embed_from_df but is used at inference time where we don't have the
# label for the sentence.
def featurize_X_from_text(text_X, wvecs):
    """
    For the given text, creates a the feature vectors (based on the embeddings provided by wvecs)

    Parameters:
        text_X: string to featurize
        wvecs (dict): dict of embeddings, but ideally should be KeyedVectors from gensim

    Return:
        featurized numpy array of shape (1, MAX_WORD_SIZE*EMBED_SIZE)
    """
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
    """
    For the given article text, creates sentences and preprocesses them.
    Usually used during inference time.

    Parameter:
        inf_art_t: article text

    Return:
        list of 2 items:
            orig_sent (list): list of actual sentences from provided text
            pre_sent (list): list of preprocessed sentences from provided text
    """
    art_splits = inf_art_t.split('\n')
    orig_sent = []
    pre_sent = []
    min_count = preprocessing_bbc.WORD_COUNT_THRES
    for split in art_splits:
        if len(split) == 0:
            continue
        art_sents = (sent_tokenize(split))
        for sent in art_sents:
            orig_sent.append(sent)
            inf_clean_t, inf_clean_count = preprocessing_bbc.apply_re(sent)
            if min_count < inf_clean_count <= MAX_WORD_SIZE:
                pre_sent.append(inf_clean_t.strip())
    return orig_sent, pre_sent