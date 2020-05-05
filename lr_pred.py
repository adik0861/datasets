from pathlib import Path
import nltk
import numpy as np
import pandas as pd
from joblib import dump, load
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import requests


class Data:
    def __init__(self):
        self.download_files()
        self.train, self.test = pd.read_csv('train.csv'), pd.read_csv('test.csv')
        self.stop_words = self.get_stopwords()
        self.vectorizer = TfidfVectorizer(min_df=5,
                                          sublinear_tf=True,
                                          strip_accents='ascii',
                                          stop_words=self.stop_words,
                                          decode_error='replace')
        self.train_vectors = self.vectorizer.fit_transform(self.train.Content)
        self.test_vectors = self.vectorizer.transform(self.test.Content)

    def vectorize(self, text):
        text = [text] if isinstance(text, str) else text
        return self.vectorizer.transform(text)

    @staticmethod
    def download_files():
        base_url = 'https://github.com/adik0861/sentiment_analysis/blob/master/'
        for _file in ['train.csv', 'test.csv']:
            if not Path(_file).exists():
                with open('train.csv', 'wb') as f:
                    f.write(requests.get(f'{base_url}{_file}').content)

    @staticmethod
    def get_stopwords():
        stop_words = list()
        if not Path.home().joinpath('nltk_data/corpora/stopwords').exists():
            nltk.download('stopwords')
        stop_words += list(stopwords.words('english'))
        stop_words += [x.replace("'", '') for x in stop_words if "'" in x]
        return stop_words


class Model(Data):
    def __init__(self):
        super().__init__()
        self.lr = self.load_model()

    def load_model(self):
        if not Path('lr_sentiment.model').exists():
            print('\nModel file not found.  Training from scratch.')
            return self.train_model()
        else:
            print('\nModel file found.  Loading model.')
            return load('lr_sentiment.model')

    def grid_search(self):
        _C = list(np.logspace(0, 4, 10)) + [10 ** x for x in range(-3, 3)]
        grid = GridSearchCV(LogisticRegression(), {'C': _C}, cv=5)
        grid.fit(self.train_vectors, self.train.Label)
        return grid.best_estimator_

    def train_model(self):
        print('Training model...\n')
        lr = self.grid_search()
        lr.fit(self.test_vectors, self.test.Label)
        dump(lr, 'lr_sentiment.model')
        lr.predict(self.test_vectors)
        _score = lr.score(self.test_vectors, self.test.Label)
        return lr

    def predict_sentiment(self, text):
        prediction = self.lr.predict(self.vectorize(text))
        print(f'Sentiment: {prediction[0]}')
        return prediction


if __name__ == '__main__':
    model = Model()
    while True:
        model.predict_sentiment(input('\nInput Text:\t'))
