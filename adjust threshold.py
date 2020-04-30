import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,zero_one_loss, roc_curve, auc, precision_recall_curve, accuracy_score
from os import chdir
from time import time

# Based on https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(p, r, thresholds, y_scores, y_test, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print('Threshold: ',t,'\n',pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.5, 1.01])
    plt.xlim([0.5, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)

    plt.show()

def train_model(train_features,train_labels):
    # Initialise classifier and train model
    start = time()
    classifier = RandomForestClassifier(n_jobs=-1,random_state=49,n_estimators=250,max_depth=32)
    print('Training Model...')
    model = classifier.fit(train_features,train_labels)
    print('Training Complete in ' + str(round(time()-start,2)) + ' seconds...')
    return model

def main():
    chdir('/home/zack/Desktop/Machine-Learning-CMP417') # Change to data directory

    train_filepath = 'AIdataERS4M/UNSW_NB15_training-set-ERS4M.csv'
    test_filepath = 'AIdataERS4M/UNSW_NB15_testing-set-ERS4M.csv'

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

    model = train_model(train_features,train_labels)
    print('Score: ', round((model.score(train_features,train_labels)*100),2),'%')

    probabilities = model.predict_proba(test_features)

    test_labels_binary = []
    label_score = []
    for n in range(len(test_labels)):
        label = test_labels[n]
        label_score.append(sum(probabilities[n,:])-probabilities[n,0]) #sum probabilities of all attack categories excluding normal
        if attacks[label] == 'Normal':
            assert (label == 0) #check that the label for normal is 0
            test_labels_binary.append(0)
        else:
            test_labels_binary.append(1)

    test_labels_binary = np.array(test_labels_binary)
    label_score = np.array(label_score)

    precision, recall, thresholds = precision_recall_curve(test_labels_binary, 
    label_score)

    precision_recall_threshold(precision,recall,thresholds,label_score, test_labels_binary)
    precision_recall_threshold(precision,recall,thresholds,label_score, test_labels_binary, t=0.25)
    precision_recall_threshold(precision,recall,thresholds,label_score, test_labels_binary, t=0.2)

if __name__ == '__main__':
    main()