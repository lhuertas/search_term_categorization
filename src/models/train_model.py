import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from text_processing import process as prepare_data
import pickle
from pathlib import Path
import argparse

def train(X, y, save_model=True):

    print("Training model with Naive Bayes...")
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                  ])
 
    nb.fit(X, y)
    print("training done...")

    # save the model to disk
    if save_model:
       filename = 'trainded_model.sav'
       print("Saving model into file: {}".format(filename))
       pickle.dump(nb, open(Path(filename), 'wb'))
    else:
       return nb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""This script will train a classifier """)
    parser.add_argument("trainSet", help="Train sample")
    args = parser.parse_args()

    df_train = pd.read_csv(args.trainSet, header=None, names=['term','category'])
    df_train = prepare_data(df_train)

    train(df_train.text_clean, df_train.category)



