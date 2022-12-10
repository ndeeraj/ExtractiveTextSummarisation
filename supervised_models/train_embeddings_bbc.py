import os
import time
import pandas as pd
import preprocessing_bbc
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

'''
This script will create the word2vec learned embeddings from the preprocessed data 
in [project-root]/generated
'''
EMBEDDINGS_SIZE = 50

curr_dir = os.getcwd()
parent = os.path.dirname(curr_dir)

embedding_file = os.path.join(parent, 'generated', 'embeddings_bbc.txt')
# not being used downstream, so not storing for now
#model_file = os.path.join(parent, 'generated', 'wrd2vec_bbc.model')

train_file = preprocessing_bbc.cleaned_train_f
test_file = preprocessing_bbc.cleaned_test_f
val_file = preprocessing_bbc.cleaned_val_f


# reads the data from the files to dataframes
def prepare_data():
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    val_data = pd.read_csv(val_file)

    data = []
    print("loading tokens from train data")
    add_data_from_df(data, train_data)
    print("loading tokens from test data")
    add_data_from_df(data, test_data)
    print("loading tokens from validation data")
    add_data_from_df(data, val_data)

    return data


def add_data_from_df(data, df):
    for i in range(len(df)):
        inp_t = df.loc[i, "input"].lower().split()
        data.append(inp_t)


def train_gensim(embed_data, embed_file, sg=1, emd_size=EMBEDDINGS_SIZE, window=5, min_count=1):
    '''
    Trains the Word2Vec model from Gensim with the provided parameters.
    Also, saves the embeddings and model to the provided files.
    '''

    model = Word2Vec(sentences=embed_data, window=window, sg=sg, min_count=min_count,
                     vector_size=emd_size, seed=42)
    model.wv.save_word2vec_format(embed_file, binary=False)

    print('Vocab size {}'.format(len(model.wv)))
    return model


# Utilities to retreive the word2vec model or embedding from the provided file
def load_gensim(model_file):
    model = Word2Vec.load(model_file)
    return model


def load_embeddings(embed_file):
    wv_from_text = KeyedVectors.load_word2vec_format(embed_file, binary=False)
    return wv_from_text


if __name__ == '__main__':
    start = time.time()
    emd_data = prepare_data()
    train_gensim(emd_data, embedding_file)
    end = time.time()
    print("Time taken: " + str(end - start))
