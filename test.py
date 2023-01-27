import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from flask import Flask, jsonify



def english_intent():
    df = pd.read_csv(r'C:\Users\RISHIKEESH\BIG\CMO\eng.csv')
    df.drop('sr.no', inplace=True, axis=1)
    asking_for_help = ['issue', 'issues', 'sir please', 'suffering', 'ask', 'help', 'support', 'please help',
                       'speak hindi', 'milk', 'product', 'ration', 'groceries', 'medical', 'medicine', 'doctor',
                       'doctors', 'emi', 'rent', 'tension',
                       'vegetable', 'karagir', 'worker', 'hospital', 'job', 'security']

    sentences = list(df["message"])
    print(sentences)
    #words = cleaning(sentences)
    #print(words)
    print('success')



def cleaning(sentences):
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        lemmatizer = WordNetLemmatizer()
        # lemmatizing
        words.append([lemmatizer.lemmatize(i.lower()) for i in w])

    return words


def test():
    from inltk.inltk import setup
    setup('hi')
    



def main():
    #english_intent()
    test()
    print('adding some changes to it')



if __name__ == '__main__':
    main()