# Extractive Text Summarisation

Perform extractive text summarization on BBC dataset using TextRank and other supervised models (logistic regression, SVM, feed forward network) and evaluate their performance based on ROUGE scores.

Text rank
-----------------
\[how to run? - FILL ME\]

Supervised Models
-----------------

The supervised models expects the preprocessed sentence level labelled train, test, validation data to be in \[project-root\]/generated-data/train_clean.csv, \[project-root\]/generated-data/test_clean.csv, \[project-root\]/generated-data/validation_clean.csv respectively and trained embeddings from the cleaned data in \[project-root\]/generated/embeddings_bbc.txt

If you want to generate these files yourself, you should use the below data pipeline:

- The BBC dataset with folder name as "bbc-data" with subfolders 'News Articles', 'Summaries' should be placed at \[project-root\]
- run `supervised_models/scripts/prepare_bbc_data.py`, this script will create train, test, validation csv files in \[project-root\]/generated-data
- run `supervised_models/scripts/generate_sent_labels.py`, this script will create sentence level labelled data in \[project-root\]/generated-data based on the files from above step
- run `supervised_models/scripts/preprocessing_bbc.py`, this script will create preprocessed data from the above labelled files in \[project-root\]/generated-data
- run `supervised_models/scripts/train_embeddings_bbc.py`, this script will create the word2vec learned embeddings from the preprocessed data in \[project-root\]/generated

Once these generated files are in place, you can follow the steps in the following notebooks to generate results from different models:

Logistic regression: `supervised_models/LogisticR.ipynb`

SVM: `supervised_models/SVM.ipynb`

Feed forward NN: `supervised_models/NN.ipynb`

These notebooks create their corresponding result csv files containing the original articles, original summary, model summary (summary generated from the trained model) for the test set which can be used to compute the ROUGE scores.

Results
-----------------

ROUGE scores can be computed using the script `[project-root]/supervised_models/rouge_results.py`, the script expects the result files from the models to be in `[project-root]/supervised_models/logr_results.csv'`, `'[project-root]/supervised_models/svm_results.csv'`, `'[project-root\]/supervised_models/NN_results.csv'` for logistic regression, svm, feed forward models respectively. This script loads the text rank results from 
`[project-root]/generated-data/RougeScoreTextRank_testset.csv` which was created upstream and plots
the performance of all the 4 models for each ROUGE metric
