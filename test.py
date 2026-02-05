#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# -- Standard library {
import decimal

import platform
import os
import os.path
from os                         import listdir
from pathlib                    import Path
from os                         import sep
from os                         import path
from os                         import walk
# from os.path                  import join <-- SQLAlchemy에서 사용
from os                         import linesep
from os.path                    import expanduser
from os.path                    import isfile

from shutil                     import copyfileobj
from shutil                     import copyfile
from subprocess                 import PIPE
from subprocess                 import Popen
import subprocess

import errno
import getopt
import glob
import sched
import warnings

import gzip
import tarfile
import zipfile
import zlib

from sys                        import argv
from sys                        import stderr
from sys                        import stdin
import sys

import io

import codecs
import bz2
import pickle
import csv
import sqlite3
import random

from statistics                 import *
"""
| * statistics.mean
| * statistics.fmean
| * statistics.geometric_mean
| * statistics.harmonic_mean
| 
| * statistics.median
| * statistics.median_grouped
| * statistics.median_high
| * statistics.median_low
| 
| * statistics.mode
| * statistics.multimode
| 
| * statistics.pstdev
| * statistics.stdev
| 
| * statistics.correlation
| 
| * statistics.variance
| * statistics.pvariance
| * statistics.covariance
| 
| * statistics.quantiles
| * statistics.NormalDist
| 
| * statistics.linear_regression
| 
|   statistics.StatisticsError
"""
import math

from time                       import sleep
import time
import datetime
import calendar

from uuid                       import uuid4
import base64
import hashlib
import json
import re
import secrets
import string
import textwrap 
import keyword
import difflib

from pprint                     import pprint
from inspect                    import stack
import inspect
import traceback
import unittest
import pdb

from collections                import Counter
from collections                import namedtuple
from functools                  import *                # lru_cache
from itertools                  import chain
from operator                   import add
import queue

from multiprocessing            import Event
from multiprocessing            import JoinableQueue    # JoinableQueue.join()
from multiprocessing            import Lock
from multiprocessing            import Pool
from multiprocessing            import Process
from multiprocessing            import Queue            # multiprocessing.Manager().Queue()|.list()|.dict() - IPC manager
from multiprocessing            import TimeoutError
from multiprocessing            import cpu_count
import multiprocessing                                  # .Manager(), .Lock() .Semaphore()
import signal
import threading

from concurrent.futures         import ProcessPoolExecutor
from concurrent.futures         import ThreadPoolExecutor
from concurrent.futures         import as_completed
from concurrent.futures         import wait

import socket

import urllib
from urllib.error               import URLError
from urllib.parse               import quote
from urllib.parse               import quote_plus
from urllib.parse               import unquote
from urllib.parse               import unquote_plus
from urllib.parse               import urlencode
from urllib.parse               import urljoin
from urllib.parse               import urlparse
from urllib.request             import Request
from urllib.request             import urlopen
from http.client                import RemoteDisconnected

from email.mime.multipart       import MIMEMultipart
from email.mime.text            import MIMEText
from smtplib                    import SMTP
from smtplib                    import SMTPAuthenticationError
from smtplib                    import SMTP_SSL

from logging.config             import dictConfig
import logging
# -- Standard library }

# -- 3rd party library::Debug {
from memory_profiler            import profile
# -- 3rd party library::Debug }

# -- 3rd party library::Graph {
from matplotlib.colors          import ListedColormap
from pandas.plotting            import register_matplotlib_converters
import matplotlib
import matplotlib.pyplot as plt
plt.rcdefaults()

# -- 3rd party library::Graph }

# -- 3rd party library::Numerical {
import numpy as np

import scipy
from scipy.stats                      import norm
from scipy                            import stats
from scipy.cluster.hierarchy          import dendrogram
from scipy.cluster.hierarchy          import linkage
# $      pip3 install --upgrade --user scipy
# $ sudo pip3 install --upgrade        scipy

import pandas as pd

from IPython.display                  import Image
# -- 3rd party library::Numerical }

# -- 3rd party library::Statistics {
# import eli5
import pycrfsuite  # ← import sklearn_crfsuite
# pip install python-crfsuite

import joblib

from io                               import StringIO
# or use `from six import StringIO` instead `sklearn.externals.six.StringIO`

from sklearn                          import datasets
from sklearn                          import linear_model
from sklearn                          import metrics
from sklearn                          import model_selection
from sklearn                          import preprocessing
from sklearn                          import svm
from sklearn.cluster                  import AgglomerativeClustering
from sklearn.cluster                  import KMeans
from sklearn.datasets                 import fetch_20newsgroups
from sklearn.datasets                 import make_blobs
from sklearn.ensemble                 import BaggingClassifier
from sklearn.ensemble                 import RandomForestClassifier
from sklearn.exceptions               import NotFittedError  # : Vocabulary not fitted or provided
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.feature_extraction.text  import HashingVectorizer
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.linear_model             import LinearRegression
from sklearn.linear_model             import LogisticRegression
from sklearn.linear_model             import SGDClassifier
from sklearn.metrics                  import accuracy_score
from sklearn.metrics                  import classification_report
from sklearn.metrics                  import confusion_matrix
from sklearn.metrics                  import mean_squared_error
from sklearn.metrics                  import precision_recall_fscore_support
from sklearn.metrics                  import r2_score
from sklearn.metrics                  import roc_auc_score
from sklearn.metrics                  import roc_curve
from sklearn.metrics.pairwise         import cosine_similarity
from sklearn.model_selection          import KFold
from sklearn.model_selection          import LeaveOneOut
from sklearn.model_selection          import LeavePOut
from sklearn.model_selection          import ShuffleSplit
from sklearn.model_selection          import StratifiedKFold
from sklearn.model_selection          import cross_val_score
from sklearn.model_selection          import train_test_split
from sklearn.naive_bayes              import GaussianNB
from sklearn.naive_bayes              import MultinomialNB
from sklearn.neighbors                import KNeighborsClassifier
from sklearn.pipeline                 import make_pipeline
from sklearn.preprocessing            import LabelBinarizer
from sklearn.preprocessing            import LabelEncoder
from sklearn.preprocessing            import PolynomialFeatures
from sklearn.preprocessing            import StandardScaler
from sklearn.svm                      import SVC
from sklearn.tree                     import DecisionTreeClassifier
from sklearn.tree                     import export_graphviz
from sklearn.tree                     import plot_tree
import sklearn  # print(sklearn.__version__) → 1.0.2

# -- 3rd party library::Statistics }

# -- 3rd party library::Language {
import nltk
from nltk                             import sent_tokenize
from nltk                             import word_tokenize
from nltk.classify                    import NaiveBayesClassifier
from nltk.corpus                      import wordnet
from nltk.tokenize                    import RegexpTokenizer
# -- 3rd party library::Language }


def pd_setup():
    pd.set_option("display.max_rows"    , 5000) # number of rows
    pd.set_option("display.max_columns" ,  500) # number of columns
    pd.set_option("display.width"       ,  168) # screen
    pd.set_option("display.max_colwidth",   21) # column
    pd.set_option("colheader_justify"   , "left")

    pd.options.display.memory_usage      = True
    pd.options.display.pprint_nest_depth = 3
    pd.options.display.precision         = 3
    pd.options.display.show_dimensions   = True
    pd.options.display.float_format      = "{:,.2f}".format
    # print(np.finfo(np.double).precision)
    # print(np.finfo(np.longdouble).precision)


def load_data():
    r'''columns
    city
    city_ascii
    lat
    lng
    country
    iso2
    iso3
    admin_name
    capital
    population
    id
    '''

    f = "/home/koowgnojeel/TODO/ml/sample-worldcities.csv"
    df = pd.read_csv(f)
    df = df.astype(
        {
             "city"        : "string"
            ,"city_ascii"  : "string"
            ,"lat"         : "float64"
            ,"lng"         : "float64"
            ,"country"     : "string"
            ,"iso2"        : "string"
            ,"iso3"        : "string"
            ,"admin_name"  : "string"
            ,"capital"     : "string"
            ,"population"  : "int64"
            ,"id"          : "int64"
        }
    )

    return df

pd_setup()
df = load_data()

# print(df.head())
# print(df.dtypes)

def scaling_transform_vectorization():
    # Scaling vs. Transformation
    # Scaling keeps the original distribution shape
    # but changes the scale.
    # Transformation changes the shape of the distribution.
    # Vectorization is typically a precursor (선구자),
    # turning raw features into numerical data,
    # which are then scaled or transformed.

    # Label encoding
    mLabelEncoder_X      = preprocessing.LabelEncoder()
    mLabelEncoder_y      = preprocessing.LabelEncoder()
    mLabelEncoder_y_uniq = preprocessing.LabelEncoder()

    # convert int to str
    D       = " / "  # delimiter
    lat     = df["lat"].to_numpy().astype(str)
    lng     = df["lng"].to_numpy().astype(str)
    city    = df["city"].to_numpy().astype(str)
    country = df["country"].to_numpy().astype(str)

    # element-wise concatenation, csv
    lat_  = np.char.add(lat, D)
    lng_  = np.char.add(lng, D)
    city_ = np.char.add(city, D)
    coordinates = np.char.add(
         np.char.add(lat_,  lng_)
        ,np.char.add(city_, country)
    )

    coordinates.shape

    y = dependent_var = coordinates


    # mLabelEncoder_y_uniq = mLabelEncoder_y_uniq.fit_transform(list(set(y)))
    mLabelEncoder_y.fit(y)
    # raise ValueError
    #     y should be a 1d array, got an array of shape () instead.
    #     y should be a 1d array, got an array of shape (106, 2) instead.


    '''stack and save as csv into buffer
    | # combine them into a single 2D array where each input array becomes a column
    | combined_array = np.column_stack((
    |      df["city"].to_numpy()
    |     ,df["country"].to_numpy()
    | ))
    | 
    | csv_buffer = io.StringIO()  # in-memory file object
    | np.savetxt(csv_buffer, combined_array, delimiter=' / ', fmt='%s')
    | csv_string = csv_buffer.getvalue()  # get the CSV content as a string
    | csv_buffer.close()
    | 
    | ... = mDF[X.split(",")].to_csv(index=False, header=False).split("\n")
    | 
    '''

    X = independent_var = city

    # Filter out empty value

    mLabelEncoder_X.fit(X)

    # save
    # joblib.dump(mLabelEncoder_y       , '/dev/shm/temp.LabelEncoder_y"))
    # # joblib.dump(mLabelEncoder_y_uniq, '/dev/shm/temp.LabelEncoder_y_uniq"))
    # joblib.dump(mLabelEncoder_X       , '/dev/shm/temp.LabelEncoder_X"))



# if mTransformer_X == "None":
#     if mAlgorithm == "Linear regression":
#     elif mAlgorithm == "Polynomial regression":
#     elif mAlgorithm == "** Hierarchical clustering(HAC)":
#     elif mAlgorithm == "** k-means":
# else:
#     if mAlgorithm == "Logistic regression":
#     elif mAlgorithm == "Support vector machine(SVM)":
#     elif mAlgorithm == "Decision tree":
#     elif mAlgorithm == "*** Bootstrap aggregation":
#     elif mAlgorithm == "*** Random forest":
#     elif mAlgorithm == "* Gaussian Naive Bayes(GNB)":
#     elif mAlgorithm == "* Multinomial Naive Bayes":
