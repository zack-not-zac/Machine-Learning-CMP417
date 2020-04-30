import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,zero_one_loss, roc_curve, auc, precision_recall_curve
from os import chdir
from time import time
from random import randint

def draw_confusion_matrix(results):
    # Visualize confusion matrix
    n_groups = results.shape[0]
    fig, ax = plt.subplots(figsize=(14,14))
    im = ax.imshow(np.log(results), cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Predicted Category') #note that the x-axis shows the columns (j) of the matrix (i,j)
    ax.set_ylabel('Actual Category') #note that the y-axis shows the rows (i) of the matrix (i,j)
    ax.set_xticks(np.arange(n_groups))
    ax.set_yticks(np.arange(n_groups))
    ax.set_xticklabels(np.arange(n_groups))
    ax.set_yticklabels(np.arange(n_groups))
    # Loop over data dimensions and create text annotations.
    for i in range(n_groups):
        for j in range(n_groups):
            text = ax.text(j, i, results[i, j], ha="center", va="center", color="w")#note that the x-axis shows the columns (j) of the matrix (i,j)
    ax.set_title('Confusion matrix for neural network')
    fig.savefig('Confusion matrix for neural network.png')
    plt.show()
    return

def draw_curves(attacks,test_labels,probabilities):
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
    fpr, tpr, thresholds = roc_curve(test_labels_binary, label_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds_precision_recall = precision_recall_curve(test_labels_binary, 
    label_score, pos_label=1)
    auc_pr = auc(recall, precision)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %0.6f' % (roc_auc))
    plt.xlabel('False Positive Rate (1-specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('Receiver Operating Characteristic (ROC) curve\n(for detecting attacks of any kind)')
    plt.legend(loc="lower right", prop={'size': 'small'})
    plt.savefig('ROC curve for neural network.png')
    plt.savefig('ROC curve for neural network.pdf')
    plt.show()
    plt.figure()
    plt.plot(recall, precision, label='AUC = %0.6f' % (auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('Precision Recall curve (for detecting attacks of any kind)')
    plt.legend(loc="lower right", prop={'size': 'small'})
    plt.savefig('Precision Recall curve for neural network.png')
    plt.savefig('Precision Recall curve for neural network.pdf')
    plt.show()
    return

def draw_recall_ROC_curves(TPR,FPR):
    return

def train_model(train_features,train_labels):
    # Initialise classifier and train model
    start = time()
    classifier = RandomForestClassifier(n_jobs=-1,random_state=randint(0,100),n_estimators=250,max_depth=32)
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
    predictions = model.predict(test_features)
    
    #Performance Metrics
    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for prediction,actual in zip(predictions,test_labels):
        if prediction == actual:
            correct += 1
        if attacks[actual] == 'Normal':             # If packet was classified Normal
            if attacks[prediction] != 'Normal':     # If packet was predicted as Attack
                FP += 1                             # Add 1 to False Positive
            else:
                TN += 1                             # Else Add 1 to True Negative
        elif attacks[actual] != 'Normal':           # If packet was classified as an attack packet
            if attacks[prediction] == 'Normal':     # If packet was predicted as a normal packet
                FN += 1                             # Add 1 to False Negative
            else:
                TP += 1                             # Else add 1 to True Positive

    print("\nConfusion Matrix for True & False Positives and Negatives")
    print("P | " + str(TP) + " |" + str(FP))
    print("N | " + str(TN) + "  |" + str(FN))
    print("--|" + "-"*(len("P | " + str(TP) + "  " + str(FP))-3))
    print("  | T      | F\n")

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)

    print('TPR: ',TPR)
    print('FPR: ',FPR) 

    accuracy = (correct/test_labels.shape[0])*100
    print('Model Accuracy Against Test Data: ', round(accuracy, 2), '%')

    results = confusion_matrix(predictions,test_labels,labels=[i for i in range(len(attacks))])
    error = zero_one_loss(test_labels, predictions)
    # By definition, entry i,j in a confusion matrix is the number of observations actually in group i, but predicted to be in group j
    print ("Error: ", error)
    draw_confusion_matrix(results)

    probabilities = model.predict_proba(test_features)
    draw_curves(attacks,test_labels,probabilities)
    
if __name__ == '__main__':
    main()