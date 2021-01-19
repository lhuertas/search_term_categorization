import os
import sys
import pandas as pd
import argparse
PROJECT_PATH = os.getcwd()
SRC_PATH = "{}/{}".format(PROJECT_PATH, "src")
sys.path.append(PROJECT_PATH)
sys.path.append(SRC_PATH)

import pickle
from models.text_processing import process as prepare_data
from pathlib import Path
import pandas as pd

def save_prediction(prediction, fileName = 'pred'):
    output = pd.DataFrame({'category': prediction})
    output.to_csv(f'{fileName}.csv')

def make_predictions(model_file, testSet):

    test = pd.read_csv(testSet, header=None, names=['term'])
    test = prepare_data(test)
    X_test = test.text_clean

    # load the model
    loaded_model = pickle.load(open(model_file, 'rb'))

    # make predictions
    y_pred = loaded_model.predict(X_test)
    
    # create the results file 
    save_prediction(y_pred, "predictions")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""This script will perform predictions and save them to a csv file """)
    parser.add_argument("model", help="Model previously trainned")
    parser.add_argument("testSet", help="Test sample to make predictions")
    args = parser.parse_args()

    trained_model = args.model
    testSet = args.testSet

    make_predictions(trained_model, testSet)
