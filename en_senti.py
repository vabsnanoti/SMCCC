# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:07:27 2020

@author: BharatAgri 

@basic structure python file

"""
from __future__ import print_function
import importlib
from distutils.version import LooseVersion
from configparser import ConfigParser
import time
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import tqdm





def dependancy_check():
    """
         checking the dependancies of packages with version

         parameter: no parameters

         returns: True: no dependancy
         false: if any dependancy

    """

    # check that all packages are installed (see requirements.txt file)
    required_packages = {
                     'pandas'
                    }

    problem_packages = list()

    for package in required_packages:
        
        try:
            p = importlib.import_module(package)        
        except ImportError:
           
            problem_packages.append(package)
    
    if len(problem_packages) is 0:
        #print('All is well.')
        return 0;
    else:
        #print('The following packages are required but not installed: ' \
          #+ ', '.join(problem_packages))
        return problem_packages;




def config_parse():

    config = ConfigParser()
    config.read('config_file.ini')
    healthy_ds = config.get('path', 'reflectance_data_healthy')
    inoculated_ds = config.get('path', 'reflectance_data_inoculated')
    return healthy_ds,inoculated_ds




    
def preprocessing():
    dict_dta={}
    with open('eng.txt', encoding='ISO-8859-2') as f:
        text = f.read()

    sent_tokenizer = PunktSentenceTokenizer(text)
    sents = sent_tokenizer.tokenize(text)

    word_tokenize(text)
    sent_tokenize(text)

    porter_stemmer = PorterStemmer()
    nltk_tokens = nltk.word_tokenize(text)

    for w in nltk_tokens:
        porter_stemmer.stem(w)

    wordnet_lemmatizer = WordNetLemmatizer()
    nltk_tokens = nltk.word_tokenize(text)

    for w in nltk_tokens:
        wordnet_lemmatizer.lemmatize(w)

    text = nltk.word_tokenize(text)
    nltk.pos_tag(text)

    sid = SentimentIntensityAnalyzer()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    compound_lst = []
    neg_lst = []
    with open('eng.txt', encoding='ISO-8859-2') as f:
        for text in f.read().split('\t'):
            #print(text)
            scores = sid.polarity_scores(text)
            for key in sorted(scores):
                #print('{0}: {1}, '.format(key, scores[key]), end='')
                print(key)



    return dict_dta


def main():
    
    print('in main')
    check = dependancy_check()
    if check == 0:
        dict_dta = preprocessing()
        #print(dict_dta)
    else:
        #load_images()
        print('The following packages are required but not installed: ' \
          + ', '.join(check))


if __name__ == '__main__':
    main()

    
    

