#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:17:10 2022

@author: skj

Description to run the file:
1. The datafile should be locally avaivable on the machine running the code. The path on lines 53 and 54 will need to be changed accodingly
1. Make sure all the libraries used are installed in the environment
2. Rouge can be installed with the command 'pip install rouge'
3. Make sure the networkx version < 2.7 (we are using 2.6.3) and the scipy version is 1.7.3 .
 If the networkx version > 2.7 or scipy > 1.8, it will make the code fail s
 ince there is a compatibity issue with scipy newer versions. Refer to the 
 stackexchance thread below:
     
https://stackoverflow.com/questions/74175462/attributeerror-module-scipy-sparse-has-no-attribute-coo-array

"""

#Import all the libraries here
import pandas as pd
import numpy as np
import os
import re 

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
#!pip install networkx
import networkx as nx
import scipy.sparse

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

#not using lemmatization for text processing, since we need grammatically correct sentences
#from nltk.stem import WordNetLemmatizer 
#lemmatizer = WordNetLemmatizer()
#from tensorflow.keras.preprocessing.text import Tokenizer 
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#stop_words = set(stopwords.words("english"))  

 #!pip install rouge 
from rouge_score import rouge_scorer
from rouge import Rouge


""" The first step is to gather all the data on text articles and their summaries
and present them in the form of a pandas dataframe for further work"""

folder_articles = 'BBC News Summary/News Articles'
folder_summaries = 'BBC News Summary/Summaries'

folder_articles_cat = os.listdir(folder_articles)

articles = []

for cat in folder_articles_cat:
    category_folder = folder_articles +'/' + cat
    #print(category_folder)
    if os.path.isdir(category_folder):
        files = os.listdir(category_folder)
        for afile in files:
            with open(category_folder +'/'+ afile, encoding= 'unicode_escape') as r:
                articles.append(r.read())
                
print("The total number of articles across all categories: ", len(articles))

folder_sum_cat = os.listdir(folder_summaries)
summaries = []

for cat in folder_sum_cat:
    category_folder = folder_summaries +'/' + cat
    #print(category_folder)
    if os.path.isdir(category_folder):
        files = os.listdir(category_folder)
        for afile in files:
            with open(category_folder +'/'+ afile, encoding= 'unicode_escape') as r:
                summaries.append(r.read())
                
print("The total number of summaries for articles across all categories: ", len(summaries))

data = {'text': articles, 'summary': summaries}
data = pd.DataFrame(data)
data.head()
print(data.shape)

""" Now that the data is neatly presented in a pandas dataframe, the next step
is to examine the data, and clean it if needed """

print("Original Article: \n" , data['text'][10])
print("Summary of the article: \n" ,data['summary'][10])

def text_cleaning(column_text_list):
    for row in column_text_list:
        
        row = re.sub("(\\t)", " ", str(row)).lower()
        row = re.sub("(\\r)", " ", str(row)).lower()
        row = re.sub("(\\n)", ".", str(row)).lower()

        # Remove _ if it occurs more than one time consecutively
        row = re.sub("(__+)", " ", str(row)).lower()

        # Remove - if it occurs more than one time consecutively
        row = re.sub("(--+)", " ", str(row)).lower()

        # Remove ~ if it occurs more than one time consecutively
        row = re.sub("(~~+)", " ", str(row)).lower()

        # Remove + if it occurs more than one time consecutively
        row = re.sub("(\+\++)", " ", str(row)).lower()

        # Remove . if it occurs more than one time consecutively
        row = re.sub("(\.\.+)", " ", str(row)).lower()

        # Remove the characters - <>()|&©ø"',;?~*!
        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", " ", str(row)).lower()

        # Remove mailto:
        row = re.sub("(mailto:)", " ", str(row)).lower()

        # Replace any url to only the domain name
        try:
            url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(row))
            repl_url = url.group(3)
            row = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(row))
        except:
            pass

        # Remove multiple spaces
        row = re.sub("(\s+)", " ", str(row)).lower()
        yield row

cleaned_text = text_cleaning(data.text)
cleaned_summary = text_cleaning(data.summary)

data['cleaned_text'] = pd.Series(cleaned_text)
data['cleaned_summary'] = pd.Series(cleaned_summary)

print(data.head())

#Exploratoty Data Analysis

article_lengths = [len(art.split()) for art in data.cleaned_text]
summary_lengths = [len(summ.split()) for summ in data.cleaned_summary]
mean_article_length = sum(article_lengths)/len(article_lengths)
mean_summary_length = sum(summary_lengths)/len(summary_lengths)

print("The average length of a textual article in the dataset is: ", mean_article_length)
print("The average length of a summary in the dataset is: ", mean_summary_length)
print("Average reduction ratio: ", mean_article_length/mean_summary_length )

print("The maximum length of an article in the datset: ", max(article_lengths))
print("The maximum length of an summary in the datset: ", max(summary_lengths))
print("The minimum length of an article in the datset: ", min(article_lengths))
print("The minimum length of an article in the datset: ", min(summary_lengths))


#Visualize the distribution of lengths of articles and summaries

plt.figure(figsize = (15,10))
plt.hist(x = article_lengths, bins = 'auto')
plt.axvline(mean_article_length, color='r', linestyle='dashed', linewidth=1)
plt.xlabel("Token counts per article")
plt.title("Distribution of token counts per article")
plt.show()

plt.figure(figsize=(15,10))
plt.hist(x = summary_lengths, bins = 'auto')
plt.axvline(mean_summary_length, color='r', linestyle='dashed', linewidth=1)
plt.xlabel("Token counts per summary")
plt.title("Distribution of token counts per summary ")
plt.show()

#Distribution looks similar

""" For the purpose of extracive text summarisation, we need to determine the 
number of sentences we shall be outputing in the summary of an article """

num_sent_articles = [len(sent_tokenize(sent)) for sent in data.cleaned_text]
num_sent_summaries = [len(sent_tokenize(sent)) for sent in data.cleaned_summary]
mean_sent_len_article = sum(num_sent_articles)/len(num_sent_articles)
mean_sent_len_summary= sum(num_sent_summaries)/len(num_sent_summaries)

print("The average number of sentences per article: ", mean_sent_len_article)
print("The average number of sentences per summary: ", mean_sent_len_summary)

print(pd.DataFrame(num_sent_articles).info())
print(pd.DataFrame(num_sent_summaries).info())
#No Nulls

print(pd.DataFrame(num_sent_articles).describe())
print(pd.DataFrame(num_sent_summaries).describe())

#Visualize the distribution of number of sentences in articles and summaries

plt.figure(figsize = (15,10))
plt.hist(x = num_sent_articles, bins = 'auto')
plt.axvline(mean_sent_len_article, color='r', linestyle='dashed', linewidth=1)
plt.xlabel("No. of sentences per article")
plt.title("Distribution of sentence length in the articles")
plt.show()

plt.figure(figsize=(15,10))
plt.hist(x = num_sent_summaries, bins = 'auto')
plt.axvline(mean_sent_len_summary, color='r', linestyle='dashed', linewidth=1)
plt.xlabel("No. of sentences per summary")
plt.title("Distribution of sentence length in the summaries")
plt.show()

"""We can start with generating 3 sentences per summary as a baseline"""



#Modeling for Extractive Text Summmarisation
""" 
The first model we shall be using is based on Google Page Rank Algorithm. 
It's called TextRank. It's a graph based model. Each sentence represents a node 
in the network graph. The weights of the connection beween two nodes/two sentences
indicate the similarity between the sentences.
Detail reading and Reference:
    
https://towardsdatascience.com/understanding-automatic-text-summarization-1-extractive-methods-8eb512b21ecc

Note: Gensim has removed the implementation of Text Rank in the new version. Hence,
we shall be implementing it for this project.

"""
#Getting pre-trained word embedding (Glove)
#Code  Reference Source: https://keras.io/examples/nlp/pretrained_word_embeddings/ 
#install and import wget in the code environment   
 
"Uncomment and run the two code lines below to download the Glove Word Embeddings"
#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip -q glove.6B.zip

word_embeddings = {}
glove_file = "glove.6B.200d.txt"

with open(glove_file) as file:
    for aline in file:
        word, coefs = aline.split(maxsplit= 1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        word_embeddings[word] = coefs
        
print("The number of words in the glove word embeddings: ", len(word_embeddings.keys()))

'''TextRank code reference: https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/'''

def TextRank(text, word_embeddings, top_n = 3):
    """

    Parameters
    ----------
    text : String
    word_embeddings : dictionary
        DESCRIPTION. Dictionay of Word embeddings (pre-trained)
    top_n : integer
        DESCRIPTION. The number of sentences to return 
    Returns
    -------
    Returns the top n sentences from the text 
    """   
    sentences_lst = sent_tokenize(text)
    
    sentence_vectors = []
    for sent in sentences_lst:
        if len(sent) != 0:
            v = sum([word_embeddings.get(w, np.zeros((200,))) for w in sent.split()])/(len(sent.split())+0.001)
        else:
            v = np.zeros((200,))
        sentence_vectors.append(v)
    
    # similarity matrix
    sim_mat = np.zeros([len(sentences_lst), len(sentences_lst)])
    
    for i in range(len(sentences_lst)):
        for j in range(len(sentences_lst)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,200), sentence_vectors[j].reshape(1,200))[0,0]
     
    #Convert the similarity matrix into a network graph, and implement PageRank on it
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    
    #Sort the nodes based on rank 
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences_lst)), reverse=True)
    
    #Return the top n ranked sentences as the summary of the text
    summ = []
    for rank in range(top_n):
        if rank < len(ranked_sentences):
            summ.append(ranked_sentences[rank][1])
        
    return " ".join(summ)

#test example
pred_sum = TextRank(data.cleaned_text[0], word_embeddings, 3)
ref_sum = data.cleaned_summary[0]
print("Summary predicted by the model: \n", pred_sum)
print("The annotated reference summary: \n", ref_sum)

gold_labels = []
predictions = []
for idx in range(len(data.cleaned_text)):
    predictions.append(TextRank(data.cleaned_text[idx], word_embeddings, 3))
    gold_labels.append(data.cleaned_summary[idx])
    
result_textrank_df = pd.DataFrame({'article' : data.text, 'ref_summary': data.summary, 'model_summary': pd.Series(predictions)})
result_textrank_df.to_csv('Results/TextRank_Results.csv')

#Model Evaluation

scorer = Rouge()

score_avg = scorer.get_scores(predictions, gold_labels, avg = True)
score_avg_df = pd.DataFrame(score_avg)
print(score_avg_df)
score_avg_df.to_csv("Results/RougeScoreTextRank_complete.csv")

#Detailed report on the score (every article)
textRank_scores = scorer.get_scores(predictions, gold_labels)
type(textRank_scores) #It's a list
len(textRank_scores) # lenght equal to the total number of articels
print(textRank_scores[0])
print(textRank_scores[0].keys())
#Pretty print it by making a dataframe

   

rouge1_recall = []
rouge1_precision = []
rouge1_f1 = []

rouge2_recall = []
rouge2_precision = []
rouge2_f1 = []

rougel_recall = []
rougel_precision = []
rougel_f1 = []


for article_score in textRank_scores:
    rouge1_recall.append(article_score['rouge-1']['r'])
    rouge1_precision.append(article_score['rouge-1']['p'])
    rouge1_f1.append(article_score['rouge-1']['f'])
    
    rouge2_recall.append(article_score['rouge-2']['r'])
    rouge2_precision.append(article_score['rouge-2']['p'])
    rouge2_f1.append(article_score['rouge-2']['f'])
    
    rougel_recall.append(article_score['rouge-l']['r'])
    rougel_precision.append(article_score['rouge-l']['p'])
    rougel_f1.append(article_score['rouge-l']['f'])
    
column_names = []
for rouge_type in textRank_scores[0].keys():
    column_names.append(str(rouge_type)+ " Recall")
    column_names.append(str(rouge_type)+ " Precision")
    column_names.append(str(rouge_type)+ " F1_score")
    
print(column_names)

scores_df = pd.DataFrame(columns = column_names)

scores_df['rouge-1 Recall'] = rouge1_recall
scores_df['rouge-1 Precision'] = rouge1_precision
scores_df['rouge-1 F1_score'] = rouge1_f1

scores_df['rouge-2 Recall'] = rouge2_recall
scores_df['rouge-2 Precision'] = rouge2_precision
scores_df['rouge-2 F1_score'] = rouge2_f1

scores_df['rouge-l Recall'] = rougel_recall
scores_df['rouge-l Precision'] = rougel_precision
scores_df['rouge-l F1_score'] = rougel_f1

print(scores_df)
print(scores_df.info())
print(scores_df.describe())

#scores_df.to_csv("Results/RougeScoreTextRank_detailed.csv")


#Evaluate the rouge scores on the common test set
test_data = pd.read_csv("bbc_combined_test.csv")
test_data.head()

cleaned_text_test = text_cleaning(test_data.article)
cleaned_summary_test = text_cleaning(test_data.summary)

test_data['cleaned_text'] = pd.Series(cleaned_text_test)
test_data['cleaned_summary'] = pd.Series(cleaned_summary_test)

testgold_labels = []
testpredictions = []
for idx in range(len(test_data.article)):
    testpredictions.append(TextRank(test_data.cleaned_text[idx], word_embeddings, 3))
    testgold_labels.append(test_data.cleaned_summary[idx])

testscore_avg = scorer.get_scores(testpredictions, testgold_labels, avg = True)
testscore_avg_df = pd.DataFrame(testscore_avg)
print(testscore_avg_df)
testscore_avg_df.to_csv("Results/RougeScoreTextRank_testset.csv")


