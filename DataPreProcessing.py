import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
np.random.seed(500)
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings("ignore")


class DataPreProcessing:

    def __init__(self):
        pass

    def prepareDataCorpus(self,fileName):

        Corpus = pd.read_json(fileName, lines=True)
        Corpus['text'].dropna(inplace=True)
        Corpus['text'] = [entry.lower() for entry in Corpus['text']]
        Corpus['text'] = [word_tokenize(entry) for entry in Corpus['text']]
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        for index, entry in enumerate(Corpus['text']):
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                    Final_words.append(word_Final)
            Corpus.loc[index, 'text_final'] = str(Final_words)
        return Corpus

    def prepareTrainTestData(self,fileName,Corpus):
        with open(fileName) as f:
            records = [json.loads(r) for r in f.readlines()]
        y = []
        for i in range(0, len(records)):
            y.append(records[i]['class'][0])
        sentences = Corpus['text_final']
        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.25, random_state=1000)
        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)

        X_train = vectorizer.transform(sentences_train)
        X_test = vectorizer.transform(sentences_test)

        return X_train,X_test,y_test,y_train

    def prepareDataForDeepModels(self,fileName):
        with open(fileName) as f:
            records = [json.loads(r) for r in f.readlines()]
        sentences = []
        y = []
        for i in range(0, len(records)):
            sentences.append(records[i]['text'])
            y.append(records[i]['class'][0])

        sentences = []
        y = []
        for i in range(0, len(records)):
            sentences.append(records[i]['text'])
            y.append(records[i]['class'][0])

        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.25, random_state=1000)

        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)
        X_train = vectorizer.transform(sentences_train)
        X_test = vectorizer.transform(sentences_test)
        from sklearn.preprocessing import LabelEncoder
        number = LabelEncoder()
        y_test = number.fit_transform(y_test)
        y_train = number.fit_transform(y_train)

        return X_train,X_test,y_test,y_train