import time
import pandas as pd
from rouge import Rouge

'''
This script computes the average ROUGE score for the result files, which are csv files with headers
'original_summary', 'model_summary'.
Assumes the result file names for logistic regression, svm, feed forward models as 
'logr_results.csv', 'svm_results.csv', 'NN_results.csv' respectively and to be in the same folder as
the script.
'''


def generate_rouge_result(result_file):
    data = pd.read_csv(result_file)
    gold_labels = []
    predictions = []
    print("generating rouge scores for results in " + result_file)
    for i in range(len(data)):
        gold_labels.append(data.at[i, 'original_summary'])
        predictions.append(data.at[i, 'model_summary'])

    scorer = Rouge()
    score_avg = scorer.get_scores(predictions, gold_labels, avg=True)
    print("average scores:")
    print(score_avg)


if __name__ == '__main__':
    start = time.time()
    res_files = ['logr_results.csv', 'svm_results.csv', 'NN_results.csv']
    for res_file in res_files:
        generate_rouge_result(res_file)
    end = time.time()
    print("Time taken: " + str(end - start))
