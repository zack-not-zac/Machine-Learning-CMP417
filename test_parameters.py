import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,zero_one_loss
from os import chdir
from time import time

def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

def train_model(train_features,train_labels):
    # Initialise classifier and train model
    # based on https://www.datasciencelearner.com/how-to-improve-accuracy-of-random-forest-classifier/
    parameters = {
        "n_estimators":[5,10,50,100,250,1000],
        "max_depth":[2,4,8,16,32,None]
    }
    classifier = RandomForestClassifier(n_jobs=-1,random_state=42)
    cv = GridSearchCV(classifier,parameters,cv=5)
    print('Testing Parameters...')
    cv.fit(train_features,train_labels)
    display(cv)

def main():
    chdir('/home/zack/Desktop/AIdataERS4M')                         # Change to data directory

    train_filepath = 'UNSW_NB15_training-set-ERS4M.csv'
    test_filepath = 'UNSW_NB15_testing-set-ERS4M.csv'

    print('Reading Data...')
    train_data = pd.read_csv(train_filepath)
    print('Train Data: {} entries with {} elements.'.format(*train_data.shape))
    test_data = pd.read_csv(test_filepath)
    print('Test Data: {} entries with {} elements.'.format(*test_data.shape))

    print('\nConverting Data...')

    train_length = train_data.shape[0]                              # Gets length of train data for later

    df = pd.concat([train_data,test_data])                          # Data is concatenated to prevent unknown protocols
    df['attack_cat'], attacks = pd.factorize(df['attack_cat'])      # Encode Attack Category
    # Split attack labels back into train and test data
    train_labels = df['attack_cat'][:train_length]                  
    test_labels = df['attack_cat'][train_length:]
    
    df = df.drop('attack_cat',axis=1)                               # Remove attack labels from Feature data
    dummies = pd.get_dummies(df)                                    # Get Dummy variables (Same as OneHotEncoders)
    
    # Split feature data back into 2 numpy arrays now with dummy variables
    train_features = dummies[:train_length].to_numpy()
    test_features = dummies[train_length:].to_numpy()

    # Print feature & label shapes for both datasets
    print('Train Features: ' + str(train_features.shape))
    print('Train Attack Labels: ' + str(train_labels.shape))
    print('Test Features: ' + str(test_features.shape))
    print('Test Attack Labels: ' + str(test_labels.shape))

    train_model(train_features,train_labels)

    # model = train_model(train_features,train_labels)
    # print('Score: ' + str(model.score(train_features,train_labels)))
    # predictions = model.predict(test_features)
    
    # #Performance Metrics
    # correct = 0
    # for prediction,actual in zip(predictions,test_labels):
    #     if prediction == actual:
    #         correct += 1

    # accuracy = (correct/test_labels.shape[0])*100
    # print('Accuracy:', round(accuracy, 2), '%.')

    # results = confusion_matrix(predictions,test_labels,labels=[i for i in range(len(attacks))])
    # error = zero_one_loss(test_labels, predictions)
    # By definition, entry i,j in a confusion matrix is the number of observations actually in group i, but predicted to be in group j
    print ("Error: ", error)
    #draw_confusion_matrix(results)
    
if __name__ == '__main__':
    main()