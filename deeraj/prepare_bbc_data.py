import time
import pandas as pd
import os

curr_dir = os.getcwd()
parent = os.path.dirname(os.path.dirname(curr_dir))


out_train_file = os.path.join(parent, 'data', 'bbc_combined_train.csv')
out_test_file = os.path.join(parent, 'data', 'bbc_combined_test.csv')
out_val_file = os.path.join(parent, 'data', 'bbc_combined_val.csv')
out_files = [out_train_file, out_test_file, out_val_file]

art_dir = os.path.join(parent, 'bbc-data', 'News Articles')
sum_dir = os.path.join(parent, 'bbc-data', 'Summaries')

art_sub_dir = os.listdir(art_dir)
sum_sub_dir = os.listdir(sum_dir)

start = time.time()

train_percent = 0.8
test_percent = 0.1
val_percent = 0.1

total = 2225


def prepare():
    headers = {'article': [],
               'summary': []}

    df_train_ind = 0
    df_test_ind = 0
    df_val_ind = 0

    combined_d_train = pd.DataFrame(headers)
    combined_d_test = pd.DataFrame(headers)
    combined_d_val = pd.DataFrame(headers)

    for topic in art_sub_dir:
        art_file_dir = os.path.join(art_dir, topic)
        sum_file_dir = os.path.join(sum_dir, topic)

        art_files = os.listdir(art_file_dir)
        sum_files = os.listdir(sum_file_dir)

        train_num = int(train_percent * len(art_files))
        test_num = train_num + int(test_percent * len(art_files))

        for i in range(len(art_files)):
            try:
                art_contents = open(art_file_dir + '\\' + art_files[i], "r",
                                    encoding='utf-8').readlines()
                sum_contents = open(sum_file_dir + '\\' + sum_files[i], "r",
                                    encoding='utf-8').readlines()
                art_content = "".join(art_contents)
                sum_content = "".join(sum_contents)

                if i < train_num:
                    combined_d_train.at[df_train_ind, 'article'] = art_content
                    combined_d_train.at[df_train_ind, 'summary'] = sum_content
                    df_train_ind += 1
                elif i < test_num:
                    combined_d_test.at[df_test_ind, 'article'] = art_content
                    combined_d_test.at[df_test_ind, 'summary'] = sum_content
                    df_test_ind += 1
                else:
                    combined_d_val.at[df_val_ind, 'article'] = art_content
                    combined_d_val.at[df_val_ind, 'summary'] = sum_content
                    df_val_ind += 1

            except UnicodeDecodeError:
                continue

    combined_d_train.to_csv(out_train_file, index=False)
    combined_d_test.to_csv(out_test_file, index=False)
    combined_d_val.to_csv(out_val_file, index=False)


if __name__ == '__main__':
    start = time.time()
    prepare()
    end = time.time()
    print("Time taken: " + str(end - start))
