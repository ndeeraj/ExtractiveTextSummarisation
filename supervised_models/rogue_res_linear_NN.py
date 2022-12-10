import time
import pandas as pd
from rouge import Rouge


def generate_rouge_result(result_file):
    data = pd.read_csv(result_file)
    data.fillna('__BLANK__')
    gold_labels = []
    predictions = []
    print("generating rouge scores for results in " + result_file)
    for i in range(len(data)):
        gold_labels.append(data.at[i, 'original_summary'])
        pred = data.at[i, 'model_summary']
        try:
            if len(pred.split()) == 0:
                pred = "__BLANK__"
        except AttributeError:
            print("----")
            print(pred)
            raise Exception()
        predictions.append(pred)

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
