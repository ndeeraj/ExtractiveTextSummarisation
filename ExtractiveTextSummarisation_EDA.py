#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:17:10 2022

@author: skj
"""

#Import all the libraries here
import pandas as pd
import numpy as np
import os
import re 

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import scipy.sparse

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

#not using lemmatization for text processing, since we need grammatically correct sentences
#from nltk.stem import WordNetLemmatizer 
#lemmatizer = WordNetLemmatizer()

from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences

#stop_words = set(stopwords.words("english"))  

  
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

        # Remove \x9* in text
        row = re.sub(r"(\\x9\d)", " ", str(row)).lower()

        # Replace INC nums to INC_NUM
        row = re.sub("([iI][nN][cC]\d+)", "INC_NUM", str(row)).lower()

        # Replace CM# and CHG# to CM_NUM
        row = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", "CM_NUM", str(row)).lower()

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




