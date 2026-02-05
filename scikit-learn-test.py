#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import asyncio
import codecs
import datetime
import errno
import getopt
import glob
import io
import json
import logging
import logging.config
import math
import os
import os.path
import pdb
import psutil
import random
import re
import signal
import socket
import socketio
import sqlite3
import ssl
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import unittest
import urllib
import urllib3
import warnings

from collections                        import namedtuple
from inspect                            import signature
from inspect                            import stack
from lxml                               import etree
from lxml                               import html
from multiprocessing                    import Event
from multiprocessing                    import Lock
from os                                 import linesep
from os                                 import path
from os                                 import sep
from os.path                            import expanduser
from pathlib                            import Path
from re                                 import search
from sys                                import argv
from sys                                import stderr
from sys                                import stdin
from textwrap                           import fill
from urllib.error                       import URLError
from urllib.parse                       import quote
from urllib.parse                       import quote_plus
from urllib.parse                       import unquote
from urllib.parse                       import urlencode
from urllib.parse                       import urljoin
from urllib.parse                       import urlparse
from urllib.request                     import Request
from urllib.request                     import urlopen
from uuid                               import uuid4

import requests
from requests.auth                      import HTTPBasicAuth

import joblib
from setproctitle                       import setproctitle

from bs4                                import BeautifulSoup


# Select independent variable X
#     city
# 
# Select dependent variable y
#     lat
#     lng
#     country
# 
# /tmp/pacemaker/model/
# 
# Platform        : Processor(x86_64), Python version(CPython 3.12.3), Platform(Linux-6.14.0-37-generic-x86_64-with-glibc2.39)
# Version         : sklearn(1.5.1), scipy(1.14.0), numpy(1.26.4), pandas(2.2.2), joblib(1.4.2)
# Number of rows  : 106
# Training time   : 0.024537014999850726
# Accuracy(0 to 1): 0.7777777777777778
# Vectorizer      : LabelEncoder(X), LabelEncoder(y)
# Algorithm       : * Gaussian Naive Bayes(GNB)
# Vectorizer files: a8e4a1e18a85c4b1524da85e01ae60bd.LabelEncoder_X, a8e4a1e18a85c4b1524da85e01ae60bd.LabelEncoder_y
# Vectorizer size : 3.22 KB
# Model file      : a8e4a1e18a85c4b1524da85e01ae60bd.GNB
# Model size      : 3.62 KB
# 


# TODO: 생성 env 기입 - version
mData      = joblib.load("/home/koowgnojeel/app/sandbox/webapp/pacemaker/app/ircclient/featuresCorrespond2X.ser" )
# column : "city"     , "lat"   ,"lng"    , (?, ?)      , ?
# value  : ('New York', '40.6943,-73.9249', (5087, 5087), 5323)

mGNBModel  = joblib.load("/home/koowgnojeel/app/sandbox/webapp/pacemaker/app/ircclient/cityname2coordinate.model")
mEstimator = joblib.load("/home/koowgnojeel/app/sandbox/webapp/pacemaker/app/ircclient/estimator.ser"            )
# mGNBModel의 city명으로 y값 lng을 찾아
# y값으로 mGNBModel에서 predict한 값으로 mEstimator에서 inverse하여 
# 실제 y값을 얻음?

mCity        = "Seoul"

# Inquiry, city name to feature value
for i in mData:
    if mCity.lower() == i[0].lower():
        mFeature = i[2]  # lng값

        y_pred = mGNBModel.predict([mFeature])
        print(y_pred)

        # # Target value 2 y value
        # y_value = mEstimator.inverse_transform(y_pred)
        # print(y_value)

        # mLat, mLon = y_value[0].split(",")
        # mTZ        = mTZWhere.tzNameAt(float(mLat), float(mLon))

        # break


