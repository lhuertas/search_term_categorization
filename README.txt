# Search Term Categorization

Train a text dataset and classify the search terms by categories.

## Packages needed

Find the list packages needed for the classfication model in the file requirements.txt


## Usage

Execute the predict.py [trained_model] [test] file to generate the classification for testset

'''bash
python predict.py src/models/trainded_model.sav data/candidateTestSet.txt 
'''

To get the trained model, please execute the code src/models/train_model.py [trainSet]
'''bash
python src/models/train_model.py data/trainSet.csv
'''

## Result

The predictions should be saved in a csv file "predictions.csv"
