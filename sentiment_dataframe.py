# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:07:27 2020

@author: BharatAgri 

@basic structure python file

"""
from __future__ import print_function
import importlib
from configparser import ConfigParser

import emoji
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import xlrd
from langdetect import detect
from tqdm import tqdm
import xlsxwriter
import json


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
        # print('All is well.')
        return 0;
    else:
        # print('The following packages are required but not installed: ' \
        # + ', '.join(problem_packages))
        return problem_packages;


def config_parse():
    config = ConfigParser()
    config.read('config_file.ini')
    healthy_ds = config.get('path', 'reflectance_data_healthy')
    inoculated_ds = config.get('path', 'reflectance_data_inoculated')
    return healthy_ds, inoculated_ds


def word_clean(s):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", str(s)).lower()
    w = word_tokenize(clean)
    # stpoword removal
    stop_words = set(stopwords.words('english'))
    w = [s for s in w if not s.lower() in stop_words]
    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(i.lower()) for i in w]


def sentiment_analysis(df):
    # df = pd.read_csv(r'C:\Users\RISHIKEESH\BIG\CMO\me_temp.csv')
    df['sep_words'] = df['message'].apply(word_clean)
    sid = SentimentIntensityAnalyzer()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    compound_lst = []
    neg_lst = []
    neu_lst = []
    pos_lst = []

    for index, row in df.iterrows():
        # print(row['message'])
        scores = sid.polarity_scores(row['message'])
        for key in sorted(scores):
            # print('{0}: {1}, '.format(key, scores[key]), end='')
            if key == 'compound':
                compound_lst.append(scores[key])
            elif key == 'neg':
                neg_lst.append(scores[key])
            elif key == 'neu':
                neu_lst.append(scores[key])
            elif key == 'pos':
                pos_lst.append(scores[key])

    dict_temp = {'compound': compound_lst, 'positive': pos_lst, 'negative': neg_lst, 'neutral': neu_lst}
    df_sent = pd.DataFrame(dict_temp)
    return df_sent


def lang_iden(df_dict):
    en_df = {}
    print('Identifying the language \n')
    for k, v in tqdm(df_dict.items()):
        langdet = []
        for i in range(len(df_dict[k])):
            try:
                lang = detect(df_dict[k].loc[i, "message"])
            except:
                lang = 'no'
            langdet.append(lang)
        df_dict[k]['language'] = pd.DataFrame(langdet)
        temp_df = df_dict[k][df_dict[k]['language'] == 'en']
        en_df.update({k: temp_df})
    return en_df


def word_clean_intent(sent):
    words = []
    for s in sent:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", str(s)).lower()
        w = word_tokenize(clean)
        # stpoword removal
        stop_words = set(stopwords.words('english'))
        w = [s for s in w if not s.lower() in stop_words]
        # lemmatizing
        lemmatizer = WordNetLemmatizer()
        words.append([lemmatizer.lemmatize(i.lower()) for i in w])

    return words



def new_intent_clean(s):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", str(s)).lower()
    clean = emoji.demojize(clean)
    return clean


def intent_analysis(df_dict):
    asking_for_help = ['issue', 'issues', 'suffering', 'ask', 'help', 'support', 'hindi', 'milk', 'product', 'ration',
                       'groceries', 'medical', 'medicine', 'doctor', 'emi', 'rent', 'tension',
                       'vegetable', 'karagir', 'worker', 'hospital', 'security', 'need', 'poor', 'garbage', 'pls',
                       'plz', 'request', 'urgent', 'safety', 'deploy', 'lock', 'lockdown', 'provide', 'please', 'look',
                       'provide', 'folded'
        , 'clinic', 'fever', 'shelter', 'favour', 'crisis', 'care', 'flight']

    prb_face = ['problem', 'ppe', 'mismanagement', 'management', 'action', 'nirnay', 'lagging', 'prashana', 'gone',
                'allow', 'raise', 'happen', 'anti', 'effort', 'dialysis', 'gathering', 'wine', 'daru', 'beer',
                'conspiracy', 'banned', 'income', 'update']

    suggest = ['suggest', 'suggestion', 'strict', 'strictly', 'shut', 'isolating', 'think', 'atleast', 'testing',
               'understand', 'properly', 'plasma', 'plan', 'refund', 'pandemonium',
               'crowd', 'decisive', 'stop']

    negative = ['fail', 'failure', 'not', 'cheap', 'garbage', 'dissatisfactory', 'careless', 'unsatisfactory',
                'horrible', 'sad', 'imperfect', 'incorrect', 'inadequate', 'defective'
        , 'wrong', 'blaming', 'failed', 'bla', 'better', 'stunt', 'pr', 'dumb', 'fake cm', 'resign', 'disgusting',
                'useless']

    appre_work = ['appreciate', 'best', 'job', 'good', 'best', 'awesome', 'proud', 'well', 'acceptable', 'excellent',
                  'marvelous', 'superb', 'jai', 'trust', 'love', 'support', 'congratulations', 'jay', 'hind',
                  'exceptional', 'favorable', 'great', 'positive', 'satisfactory', 'valuable', 'wonderful', 'nyc',
                  'nice', 'pleasing', 'grt', 'gr8', 'gud', 'supr', 'sup4', '1stclass', 'awsm', 'luv', 'agreeble',
                  'keepup'
        , 'hats', 'salute', 'spirit', 'safe', 'deserving', 'brave', 'garet', 'super', 'fight', 'thumb', 'victory',
                  'inspiration', 'hat', 'winner', 'win', 'god', 'no1', 'assal', 'fan', 'leader', 'fortunate',
                  'successful', 'dynamic', 'powerful', 'motivational', 'boss', 'bless', 'superlike', 'grate',
                  'sensible']

    will_work = ['willing', 'ready', 'ex', 'consultant', 'interest', 'arrange', 'consult', 'join'
            , 'experience']

    print("performing intent analysis\n")
    for k, v in df_dict.items():
        sent = df_dict[k]['message']
        words = word_clean_intent(sent)

        # intent class creation
        help_cat = []
        prb_cat = []
        appre_cat = []
        will_cat = []
        suggest_cat = []
        negative_cat = []

        for idx, val in enumerate(words):
            temp = set(val)
            for i in asking_for_help:
                if i in temp:
                    help_cat.append(idx)

            for i in prb_face:
                if i in temp:
                    prb_cat.append(idx)

            for i in appre_work:
                if i in temp:
                    appre_cat.append(idx)

            for i in will_work:
                if i in temp:
                    will_cat.append(idx)

            for i in suggest:
                if i in temp:
                    suggest_cat.append(idx)

            for i in negative:
                if i in temp:
                    negative_cat.append(idx)

        help_cat = set(help_cat)
        prb_cat = set(prb_cat)
        appre_cat = set(appre_cat)
        will_cat = set(will_cat)
        suggest_cat = set(suggest_cat)
        negative_cat = set(negative_cat)

        # creating the classes of the intent
        for i in help_cat:
            df_dict[k].loc[i, 'intent'] = 'asking_help'

        for i in prb_cat:
            df_dict[k].loc[i, 'intent'] = 'problem_faced'

        for i in appre_cat:
            df_dict[k].loc[i, 'intent'] = 'appreciate_work'

        for i in will_cat:
            df_dict[k].loc[i, 'intent'] = 'willing_to_help'

        for i in suggest_cat:
            df_dict[k].loc[i, 'intent'] = 'suggestion'

        for i in negative_cat:
            df_dict[k].loc[i, 'intent'] = 'negative_comment'

        df_dict[k]['intent'] = df_dict[k]['intent'].fillna('miscellaneous_msg')

    return df_dict




def new_intent(old_df_dict):
    df_dict={}
    xls = xlrd.open_workbook('vdo_intent_all.xlsx', on_demand=True)
    sheets = xls.sheet_names()

    # reading the data
    print('loading intent data')
    for i in tqdm(sheets):
        df_dict.update({i.split('.', 1)[0]: pd.read_excel('vdo_intent_all.xlsx', sheet_name=i)})

    for k, v in df_dict.items():
        df_dict[k]['message_length'] = df_dict[k]['message'].str.len()
        df_dict[k]['id.1'] = old_df_dict[k]['id.1']
        df_dict[k].dropna(inplace=True)

        '''
        for k ,v in df_dict.items():
            print(df_dict[k].shape)
            df_dict[k]['message'] = df_dict[k]['message'].apply(new_intent_clean)
            df_dict[k]['message'].dropna(inplace=True)

        return df_dict
        '''


    return df_dict




def data_processing():
    filename = 'vdo_comments.xlsx'
    df_dict = {}
    df_sent = {}

    xls = xlrd.open_workbook(filename, on_demand=True)
    sheets = xls.sheet_names()

    # reading the data
    print('Reading Data')
    for i in tqdm(sheets):
        df_dict.update({i.split('.', 1)[0]: pd.read_excel(filename, sheet_name=i)})

    # deleting unused columns
    print('Deleting the unused column \n')
    for k, v in tqdm(df_dict.items()):
        df_dict[k].drop(['path', 'id', 'parent_id', 'level', 'object_id', 'object_type', 'query_status', 'query_type',
                         'query_status', 'query_time', 'description', 'from.name', 'from.id', 'length', 'name'],
                        axis='columns', inplace=True)
        df_dict[k]['message_length'] = df_dict[k]['message'].str.len()

    df_dict = lang_iden(df_dict)

    print('creating sentiment score \n')
    for k, v in tqdm(df_dict.items()):
        df_sent.update({k: sentiment_analysis(df_dict[k])})

    #df_dict = intent_analysis(df_dict)
    df_dict = new_intent(df_dict)

    return df_dict, df_sent



def percentage_match(mainvalue, comparevalue):
    if mainvalue >= comparevalue:
        matched_less = mainvalue - comparevalue
        no_percentage_matched = round(100 - matched_less * 100.0 / mainvalue, 2)
        return no_percentage_matched
    else:
        print('please checkout your value')


def json_data_gen(df_dict, df_sent):
    json_dict = {}
    new_dict = {}
    print('Data extraction process')
    for k, v in df_dict.items():
        # positive and negative comment count

        p = list(df_sent[k][df_sent[k]['compound'] > 0].count())
        pos = p[0]
        n = list(df_sent[k][df_sent[k]['compound'] < 0].count())
        neg = n[0]

        # percentage count of categories
        values = df_dict[k]['intent'].value_counts(dropna=False).keys().tolist()
        counts = df_dict[k]['intent'].value_counts(dropna=False).tolist()
        value_dict = dict(zip(values, counts))

        for r, u in value_dict.items():
            new_dict.update({r: percentage_match(df_dict[k].shape[0], u)})

        # message separation for the 1.asking help 2. willing to help

        ask = df_dict[k][df_dict[k]['intent'] == 'asking_help']
        will = df_dict[k][df_dict[k]['intent'] == 'willing_to_help']
        comments = pd.concat([ask, will])
        comments = comments.sort_values('message_length',ascending=False)

        comments.insert(0, 'id', range(0, 0 + len(comments)))
        #comments.drop(['created_time', 'like_count', 'message_length', 'language', 'sep_words'], axis='columns',inplace=True)
        comments.drop(['message_length'], axis='columns', inplace=True)
        comments.rename(columns={'id.1': 'user_id'}, inplace=True)
        comm_j = comments.to_json(orient='records')

        json_dict.update(
            {k: {'sentiment': {'positive': pos, 'negative': neg}, 'piechart': new_dict, 'comments': comm_j}})

    return json_dict


def main():
    print('in main')
    check = dependancy_check()
    if check == 0:
        df_dict, df_sent = data_processing()

        json_dict = json_data_gen(df_dict, df_sent)

        with open('json_dict.json', 'w') as fp:
            json.dump(json_dict, fp)

        print('successfully completed')



    else:
        print('The following packages are required but not installed: ' \
              + ', '.join(check))


if __name__ == '__main__':
    main()
