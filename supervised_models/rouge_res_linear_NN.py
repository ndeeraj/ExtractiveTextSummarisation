import time
import pandas as pd
from rouge import Rouge
import matplotlib.pyplot as plt

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
    return score_avg


def visualize_res(com_res, labels):
    """
    Visualizes results in the provided dictionary

    Parameter:
        com_res (dict):  should have the below structure, and the size of each list should be equal
                        to the size of labels list passed.
                            {'rouge-1': {'r': [], 'p': [], 'f': []},
                            'rouge-2': {'r': [], 'p': [], 'f': []},
                            'rouge-l': {'r': [], 'p': [], 'f': []}}
        labels (list): list of labels for the x-axis
    """
    x_axis = labels

    '''plt.plot(x_axis, com_res['rouge-1']['r'], linestyle='dashed', label='rouge-1_r', color='red')
    plt.plot(x_axis, com_res['rouge-1']['p'], linestyle='dashed', label='rouge-1_p', color='green')
    plt.plot(x_axis, com_res['rouge-1']['f'], linestyle='dashed', label='rouge-1_f', color='black')
    plt.xlabel('methods')
    plt.ylabel('average value')
    plt.legend(loc='lower right')
    plt.show()'''

    plt.figure(1)
    plt.title("rouge-1")
    plt.plot(x_axis, com_res['rouge-1']['r'], linestyle='dashed', label='rouge-1_r', color='red')
    plt.plot(x_axis, com_res['rouge-1']['p'], linestyle='dashed', label='rouge-1_p', color='green')
    plt.plot(x_axis, com_res['rouge-1']['f'], linestyle='dashed', label='rouge-1_f', color='black')
    plt.xlabel('methods')
    plt.ylabel('average value')
    plt.ylim(0, 1)
    plt.legend(loc='best')

    plt.figure(2)
    plt.title("rouge-2")
    plt.plot(x_axis, com_res['rouge-2']['r'], linestyle='dotted', label='rouge-2_r', color='red')
    plt.plot(x_axis, com_res['rouge-2']['p'], linestyle='dotted', label='rouge-2_p', color='green')
    plt.plot(x_axis, com_res['rouge-2']['f'], linestyle='dotted', label='rouge-2_f', color='black')
    plt.xlabel('methods')
    plt.ylabel('average value')
    plt.ylim(0, 1)
    plt.legend(loc='best')

    plt.figure(3)
    plt.title("rouge-l")
    plt.plot(x_axis, com_res['rouge-l']['r'], label='rouge-l_r', color='red')
    plt.plot(x_axis, com_res['rouge-l']['p'], label='rouge-l_p', color='green')
    plt.plot(x_axis, com_res['rouge-l']['f'], label='rouge-l_f', color='black')
    plt.xlabel('methods')
    plt.ylabel('average value')
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    start = time.time()
    res_files = ['logr_results.csv', 'svm_results.csv', 'NN_results.csv']
    com_res = {'rouge-1': {'r': [], 'p': [], 'f': []},
               'rouge-2': {'r': [], 'p': [], 'f': []},
               'rouge-l': {'r': [], 'p': [], 'f': []}}

    for res_file in res_files:
        res = generate_rouge_result(res_file)
        com_res['rouge-1']['r'].append(res['rouge-1']['r'])
        com_res['rouge-1']['p'].append(res['rouge-1']['p'])
        com_res['rouge-1']['f'].append(res['rouge-1']['f'])

        com_res['rouge-2']['r'].append(res['rouge-2']['r'])
        com_res['rouge-2']['p'].append(res['rouge-2']['p'])
        com_res['rouge-2']['f'].append(res['rouge-2']['f'])

        com_res['rouge-l']['r'].append(res['rouge-l']['r'])
        com_res['rouge-l']['p'].append(res['rouge-l']['p'])
        com_res['rouge-l']['f'].append(res['rouge-l']['f'])

    visualize_res(com_res, ['logistic', 'svm', 'NN'])

    end = time.time()
    print("Time taken: " + str(end - start))
