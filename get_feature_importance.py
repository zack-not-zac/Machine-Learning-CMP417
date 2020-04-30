import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,zero_one_loss
from os import chdir

def main():
    chdir('/home/zack/Desktop/Machine-Learning-CMP417')                         # Change to data directory

    print('Reading Data...')
    train_data = pd.read_csv('AIdataERS4M/UNSW_NB15_training-set-ERS4M.csv')
    print('Train Data: {} entries with {} elements.'.format(*train_data.shape))
    test_data = pd.read_csv('AIdataERS4M/UNSW_NB15_testing-set-ERS4M.csv')
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
    feature_list = list(df.columns)
    
    # Split feature data back into 2 numpy arrays now with dummy variables
    train_features = dummies[:train_length].to_numpy()
    test_features = dummies[train_length:].to_numpy()

    # Print feature & label shapes for both datasets
    print('Train Features: ' + str(train_features.shape))
    print('Train Attack Labels: ' + str(train_labels.shape))
    print('Test Features: ' + str(test_features.shape))
    print('Test Attack Labels: ' + str(test_labels.shape))

    # Initialise classifier and train model
    classifier = RandomForestClassifier(n_jobs=-1,random_state=49,n_estimators=250,max_depth=32)
    print('Training Model...')
    model = classifier.fit(train_features,train_labels)
    print('Score: ' + str(model.score(train_features,train_labels)))
    predictions = model.predict(test_features)
    
    #Performance Metrics
    correct = 0
    for prediction,actual in zip(predictions,test_labels):
        if prediction == actual:
            correct += 1

    accuracy = (correct/test_labels.shape[0])*100
    print('Accuracy:', round(accuracy, 2), '%.')

    #Based on https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd
    # Get numerical feature importances
    importances = list(model.feature_importances_)                 # List of tuples with variable and importance
    # Sort the feature importances by most important first
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)# Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    
if __name__ == '__main__':
    main()