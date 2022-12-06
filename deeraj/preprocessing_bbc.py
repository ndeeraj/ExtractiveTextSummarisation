import re
import string
import time
import warnings

import pandas as pd
import os
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import generate_sent_labels

curr_dir = os.getcwd()
parent = os.path.dirname(os.path.dirname(curr_dir))

train_file = generate_sent_labels.train_label_file
test_file = generate_sent_labels.test_label_file
val_file = generate_sent_labels.val_label_file

cleaned_train_f = os.path.join(parent, 'data', 'train_clean.csv')
cleaned_test_f = os.path.join(parent, 'data', 'test_clean.csv')
cleaned_val_f = os.path.join(parent, 'data', 'validation_clean.csv')

WORD_COUNT_THRES = 3
MAX_THRESH = 65

vocabulary = {}


def preprocess():
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    val_data = pd.read_csv(val_file)

    print("shape of train data: " + str(train_data.shape))
    print("shape of test data: " + str(test_data.shape))
    print("shape of validation data: " + str(val_data.shape))

    trunc_d = [train_data, test_data, val_data]

    headers = {'input': [],
               'label': [],
               'word_count': [],
               }

    cleaned_train_d = pd.DataFrame(headers)
    cleaned_test_d = pd.DataFrame(headers)
    cleaned_val_d = pd.DataFrame(headers)

    clean = [cleaned_train_d, cleaned_test_d, cleaned_val_d]

    clean_text(trunc_d, clean)
    cleaned_train_d.to_csv(cleaned_train_f, index=False)
    cleaned_test_d.to_csv(cleaned_test_f, index=False)
    cleaned_val_d.to_csv(cleaned_val_f, index=False)

    print(cleaned_train_d.head(30))
    print(cleaned_test_d.head(30))
    print(cleaned_val_d.head(30))
    print("length of vocabulary: " + str(len(vocabulary)))

    return cleaned_train_d, cleaned_test_d, cleaned_val_d


def clean_text(trunc_d, clean):
    ret = []
    inp_word_count = []
    for i in range(len(trunc_d)):
        for j in range(len(trunc_d[i])):
            inp_t = trunc_d[i].loc[j, "input"]

            inp_clean_t, inp_clean_count = apply_re(inp_t)
            if WORD_COUNT_THRES < inp_clean_count <= MAX_THRESH:
                clean[i].at[j, 'input'] = inp_clean_t.strip()
                inp_word_count.append(inp_clean_count)
                clean[i].at[j, 'label'] = trunc_d[i].loc[j, "label"]
                clean[i].at[j, 'word_count'] = inp_clean_count
            else:
                if trunc_d[i].loc[j, "label"] == 1:
                    warnings.warn("rejecting text with gold true label.\nlength: " + str(inp_clean_count))

        ret.append(clean[i])

    length_df = pd.DataFrame({'word counts':inp_word_count})
    length_df.hist(bins=30)
    plt.savefig('word count for bbc corpus.png', format='png', dpi=150, bbox_inches='tight')
    plt.show()

    return ret


def apply_re(text):
    text = text.lower()
    text = re.sub('[^' + string.printable + ']', '', text)
    text = re.sub('[^a-zA-Z0-9\.\$\_\,\-]', ' ', text)
    text = re.sub('\s+', ' ', text)
    words = word_tokenize(text)
    final_words = []
    for word in words:
        if not re.fullmatch('^[' + re.escape(string.punctuation) + ']+$', word) and not len(word) == 1:
            final_words.append(word)
            vocabulary[word] = 1
    w_len = len(final_words)
    text = " ".join(final_words)
    return text, w_len


if __name__ == '__main__':
    start = time.time()
    preprocess()
    end = time.time()
    print("Time taken: " + str(end - start))
