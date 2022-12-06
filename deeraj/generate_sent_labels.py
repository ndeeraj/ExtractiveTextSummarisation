import re
import string
import time

import pandas
import pandas as pd
import os
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import prepare_bbc_data

inp_train_file = prepare_bbc_data.out_train_file
inp_test_file = prepare_bbc_data.out_test_file
inp_val_file = prepare_bbc_data.out_val_file

content_train_df = pandas.read_csv(inp_train_file)
content_test_df = pandas.read_csv(inp_test_file)
content_val_df = pandas.read_csv(inp_val_file)

WRD_LEN_THRESHOLD = 5

curr_dir = os.getcwd()
parent = os.path.dirname(os.path.dirname(curr_dir))

train_label_file = os.path.join(parent, 'data', 'bbc_labelled_train.csv')
test_label_file = os.path.join(parent, 'data', 'bbc_labelled_test.csv')
val_label_file = os.path.join(parent, 'data', 'bbc_labelled_val.csv')


def gen_sen_labels():
    headers = {'input': [],
               'label': [],
               'word_len': []}
    label_train_df = pandas.DataFrame(headers)
    label_test_df = pandas.DataFrame(headers)
    label_val_df = pandas.DataFrame(headers)

    word_lengths = []

    fill_df(content_train_df, label_train_df, word_lengths)
    fill_df(content_test_df, label_test_df, word_lengths)
    fill_df(content_val_df, label_val_df, word_lengths)

    label_train_df.to_csv(train_label_file, index=False)
    label_test_df.to_csv(test_label_file, index=False)
    label_val_df.to_csv(val_label_file, index=False)

    length_df = pd.DataFrame({'word lengths': word_lengths})
    length_df.hist(bins=30)
    plt.show()


def fill_df(content_df, label_df, word_lengths):
    label_df_ind = 0
    for i in range(len(content_df)):
        art_txt = content_df.at[i, 'article'].strip()
        sum_txt = content_df.at[i, 'summary'].strip()

        art_sents = art_txt.split('\n')
        art_sent = []
        # print(art_txt)
        # print(sum_txt)

        for tmp_txt in art_sents:
            if len(tmp_txt.strip()) < 15:
                continue
            art_sent.extend(sent_tokenize(tmp_txt))

        for art_st in art_sent:
            # print(art_st)
            if re.search(re.escape(art_st), sum_txt) is not None:
                label = 1
            else:
                label = 0
            word_count = len(art_st.split())
            if word_count <= WRD_LEN_THRESHOLD:
                continue

            label_df.at[label_df_ind, 'input'] = art_st
            label_df.at[label_df_ind, 'label'] = label
            label_df.at[label_df_ind, 'word_len'] = word_count
            label_df_ind += 1
            word_lengths.append(word_count)


if __name__ == '__main__':
    start = time.time()
    gen_sen_labels()
    end = time.time()
    print("Time taken: " + str(end - start))
