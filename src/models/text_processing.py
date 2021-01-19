import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import pandas as pd

nlp = English()

def clean_text_scipy(text):
    ''' 
	delete stopwords from string text
        implement lemmatization
    '''
    text = nlp(text)
    text = ' '.join(str(word.lemma_) for word in text if not word.is_stop) 
    return text


def get_feature(df) :
    df['word_count'] = df['term'].apply(lambda x : len(x.split()))
    df['char_count'] = df['term'].apply(lambda x : len(x.replace(" ","")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    df['total_length'] = df['term'].apply(len)
    df['num_symbols'] = df['term'].apply(lambda x: sum(x.count(w) for w in '?^!*&$%'))
    df['num_unique_words'] = df['term'].apply(lambda x: len(set(w for w in x.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['word_count']
    return df


def process(df, add_features=False):
    df['text_clean'] = df['term'].apply(clean_text_scipy)
   
    if add_features:
       df = get_feature(df)

    return df

