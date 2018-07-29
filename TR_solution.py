#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 21:49:53 2018

@author: soojunghong

@reference : 
 https://www.analyticsvidhya.com/blog/2015/09/perfect-build-predictive-model-10-minutes/

[1] Descriptive analysis on the Data – 50% time
[2] Data treatment (Missing value and outlier fixing) – 40% time
[3] Data Modelling – 4% time
[4] Estimation of performance – 6% time

@ToDo : using end-to-end project build
    https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb

@ ML project checklist
    Frame the problem and look at the big picture.

    Get the data.

    Explore the data to gain insights.

    Prepare the data to better expose the underlying data patterns to Machine Learning algorithms.

    Explore many different models and short-list the best ones.

    Fine-tune your models and combine them into a great solution.

    Present your solution.

    Launch, monitor, and maintain your system.

@ Must check the checklist for each steps
https://www.safaribooksonline.com/library/view/hands-on-machine-learning/9781491962282/app02.html#project_checklist_appendix

"""

#-----------------------------------------------------------
# 1. Frame the problem - what is the goal of given problem 
#
#   Define the objective in business terms.
#   How should you frame this problem (supervised/unsupervised, online/offline, etc.)?
#   How should performance be measured?
#   How would you solve the problem manually?
#   List the assumptions you (or others) have made so far.
#   Verify assumptions if possible.
#-----------------------------------------------------------
# learn about trending topics and concepts 
# prepare data and initial analysis
# output : give csv file and return trending topics 

#------------------------------------------------------------------------------------
# 2. Get the data 
#
#   List the data you need and how much you need.
#   Find and document where you can get that data.
#   Check how much space it will take.
#   Create a workspace (with enough storage space).
#   Get the data.
#   Convert the data to a format you can easily manipulate (without changing the data itself).
#   Ensure sensitive information is deleted or protected (e.g., anonymized).
#   Check the size and type of data (time series, sample, geographical, etc.).
#   Sample a test set, put it aside, and never look at it (no data snooping!).
#-------------------------------------------------------------------------------------
import os
import pandas as pd

CSV_PATH = "/Users/soojunghong/Documents/2018 Job Applications/ThomsonReuters_DataScientist/problem_solving/Task/"   

def load_data(csv_path=CSV_PATH):
    file_path = os.path.join(csv_path, "rna002_RTRS_2013_06.csv")
    return pd.read_csv(file_path)

data = load_data()
type(data)
data.shape #421993, 19
data.head(10)
data.describe()

#TODO : make get_data() function


#-------------------------------------------------------------------------------------
# 3. Explore data
#
#   Create a copy of the data for exploration (sampling it down to a manageable size if necessary).
#   Study each attribute and its characteristics:
#   - Name
#   - Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
#   - % of missing values
#   - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
#   - Possibly useful for the task?
#   - Type of distribution (Gaussian, uniform, logarithmic, etc.)
#   For supervised learning tasks, identify the target attribute(s).
#   Visualize the data.
#   Study the correlations between attributes.
#   Study how you would solve the problem manually.
#   Identify the promising transformations you may want to apply.
#   Identify extra data that would be useful (go back to “Get the Data”).
#   Document what you have learned.
#-----------------------------------------------------------------------------------------

# algorithm implementation
# https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

#-----------------
# data analysis 
data.head(1)
type(data["DATE"][0]) #2013-06-01 ~ 2013-06-30 -> str

type(data["STORY_DATE_TIME"][0])
data["UNIQUE_STORY_INDEX"] # DATE+PNAC
data["EVENT_TYPE"] # STORY_TAKE_OVERWRITE seems older date, DELETE has no HEADLINE_ALERT_TEXT
data["PNAC"]
data["TOPICS"] # seems the composition of all topics - multiple topics - need mapping to dictionary? 

data["NAMED_ITEMS"] #??? don't know what

data["HEADLINE_ALERT_TEXT"]

data["ACCUMULATED_STORY_TEXT"] # this fields contains some text - but mostly NaN

data["TAKE_TEXT"][421990] # may contain some text 50% take_text are missing

data["PRODUCTS"]

data["RELATED_RICS"] #??? don't know what

data["HEADLINE_SUBTYPE"] #??? don't know what

data["STORY_TYPE"] # S or NaN

data["TABULAR_FLAG"] #mostly False, some NaN and few True

data["ATTRIBUTION"] #mostly RTRS - Reuters and a few NaN 
 
data["LANGUAGE"]

#----------------------------------------
# most relevant columns for trend topic 

data["TOPICS"] # contains meaning and priority - MNGISS BACT MET BMAT MIN MINE MTAL CMPNY CA AM 






#---------------------------------------------------------------------------------
# 4. Prepare the data
#
#   Work on copies of the data (keep the original dataset intact).
#   Write functions for all data transformations you apply, for five reasons:
#   To clean and prepare the test set
#   To clean and prepare new data instances once your solution is live
#   To make it easy to treat your preparation choices as hyperparameters
# 4.1 Data Cleaning - remove outlier, filing in missing values or drop rows 
# 4.2 Feature Selection - drop attributes which don't deliver meaningful information
# 4.3 Feature Engineering - discretize continuous features, decompose features, do transformation ( (e.g., log(x), sqrt(x), x2, etc.).), aggregate features into new features
# 4.4 Feature Scaling - standardize or normalize
#----------------------------------------------------------------------------------

#--------------------------------------
# Check how many NaN values in data  
def num_missing(x): # count number of missing value in dataframe
    return sum(x.isnull())

data.apply(num_missing, axis=0)

# How to handle foreign language? Interpret to english? 


#-----------------------------------------------------------------------------
# 5. Short list of promising model 
#
#   If the data is huge, you may want to sample smaller training sets so you can train many different models
#   Train many quick and dirty models from different categories (e.g., linear, naive Bayes, SVM, Random Forests, neural net, etc.) using standard parameters.
#   Measure and compare the performance using k-fold cross-validation, and compute the mean and standard deviation of the performance measure on the N folds.
#   Analyze most significant variables for each algorithm
#   Analyze the type of errors the model make
#   Have a quick iteration of the previous steps 
#   Start with simple model and check the accuracy, precision & recall
#-------------------------------------------------------------------------------

#------------
# for data["TOPICS"] --> use hightest frequency 
data_topics = data["TOPICS"]
data_topics
type(data_topics) #Series
type(data_topics[0]) #str - 'MNGISS BACT MET BMAT MIN MINE MTAL CMPNY CA AMERS MEMI BLR LEN RTRS'


#-----------
# create set 
#Days=set(["Mon","Tue","Wed","Thu","Fri","Sat"])

from sets import Set

topic_set = set([])
topic_set
# missing TOPICS                     42738 / 421993
# --> maybe we don't need to think missing topic

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
#import mpld3

"""
def tokenize_str(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.upper() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.isalpha()]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
"""

import re
 
def str2token(topics):
    # split based on words only
    words = re.split(r'\W+', topics)
    return words

#tokenize_str("DBT GVD T24 FIN INT ISU JP JDOM JLN MOF REP FINS MCE CEN MEVN ASIA LJA RTRS")
str2token("DBT GVD T24 FIN INT ISU JP JDOM JLN MOF REP FINS MCE CEN MEVN ASIA LJA RTRS")

#---------------------------------
# for all article collect topics 
import string 
topic_set = set([])
topic_set

len(data_topics)
for idx in range(len(data_topics)): 
    arti = data_topics[idx]
    #print(arti)
    if(isinstance(arti,str)):   #data_topics[0])
        for tp in str2token(arti): 
            topic_set.add(tp)    

len(topic_set) # 1288 topics


#----------------------
# most common topic
overlap_topics = list()
for idx in range(len(data_topics)): 
    arti = data_topics[idx]
    #print(arti)
    if(isinstance(arti,str)):   #data_topics[0])
        for tp in str2token(arti): 
            overlap_topics.append(tp)    

overlap_topics
len(overlap_topics) #5765944


def getTenFreqTopics(data_topics):#(topic_series):
    overlap_topics = list()
    for idx in range(len(data_topics)):
        arti = data_topics[idx]
        #print(arti)
        if(isinstance(arti,str)):   #data_topics[0])
            for tp in str2token(arti):
                overlap_topics.append(tp)   

    overlap_topics
   
    from collections import Counter
    cnt = Counter(overlap_topics)
    #type(cnt)

    #overlap_topics.value_counts()

    print '10 Most common:'
    for letter, count in cnt.most_common(10):
        print '%s: %7d' % (letter, count)

    #type(cnt.most_common(10))
   
    return cnt.most_common(10)
        
    
    
""" 
from collections import Counter
cnt = Counter(overlap_topics)
type(cnt)

overlap_topics.value_counts()

print '10 Most common:'
for letter, count in cnt.most_common(10):
    print '%s: %7d' % (letter, count)

type(cnt.most_common(10))
"""

#-----------------------------------------
# show with bar chart most common topic
import numpy as np                                                               
import matplotlib.pyplot as plt
top_ten = cnt.most_common(15)
type(top_ten) # list
labels, ys = zip(*top_ten)
xs = np.arange(len(labels)) 
width = 0.5# 1
plt.figure(figsize=(8,8))
plt.bar(xs, ys, width, align='center')

plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels
plt.yticks(ys)


def showTenFreqTopics(top_ten):
#   top_ten = cnt.most_common(15)
 #  type(top_ten) # list
    labels, ys = zip(*top_ten)
    xs = np.arange(len(labels)) 
    width = 0.5# 1
    plt.figure(figsize=(8,8))
    plt.bar(xs, ys, width, align='center')

    plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels
    plt.yticks(ys)

#--------------------
# least common 
least_common = cnt.most_common()[-1]
type(least_common) #tuple
least_common = list()
print '10 Least common:'
for i in range(1,20):
    n = (-1)*i
#    print n
 #   for l in cnt.most_common()[n]:
    least_common.append(cnt.most_common()[n])
#        print '%s: %7d' % (letter, count)
 
least_common


#-----------------------------------------
# show with bar chart most common topic
import numpy as np                                                               
import matplotlib.pyplot as plt
type(least_common) # list
labels, ys = zip(*least_common)
xs = np.arange(len(labels)) 
width = 0.5# 1
plt.figure(figsize=(16,8))
plt.bar(xs, ys, width, align='center')

plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels
plt.yticks(ys)


#------------
# given time start date and end date, find most frequent topic
#
oneday = data.loc[(data["DATE"] == "2013-06-02")]
oneday.head(5)
oneday.shape
type(oneday)
type(oneday["TOPICS"])

oneday_topics = oneday["TOPICS"]
oneday_topics
topten = getTenFreqTopics(oneday_topics)

showTenFreqTopics(topten)

top_all_data = data["TOPICS"]
toptend_for_all = getTenFreqTopics(top_all_data)


from datetime import datetime
date = datetime.strptime(oneday["DATE"][0], '%b %d %Y %I:%M%p')
print type(date)
print date

def makeDateRange(year, month, startday, endday):
    dates = list()
    for i in range(startday, endday):
        date = str(year)+"-0"+str(month)+"-0"+str(i) # TODO : make proper 
        dates.append(date)
    return dates

period = makeDateRange(2013,06,01,05)

    

#------------
# for following, LDA and extract topic 
#data["HEADLINE_ALERT_TEXT"]

#data["ACCUMULATED_STORY_TEXT"] # this fields contains some text - but mostly NaN

#data["TAKE_TEXT"][421990] # may contain some text 50% take_text are missing


#--------------
# per date find clusters - find top 3 clusters 


#----------------------------------------------------------------------------------
# 6. Fine tune the system
#   Fine tune the hyperparameters using cross-validation 
#   Treat your transformation choices as hyperparameters - for example, replacing medien with mean, etc
#   Try Ensemble method - combine your best model often perform better
#----------------------------------------------------------------------------------


#-------------------------------------------
# 7. Present your result
#-------------------------------------------