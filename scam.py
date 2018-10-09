# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:18:11 2018

@author: Pulkit
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import preprocessing 
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

classifier_dict = {
    'Decision Tree' : [],
    'Naive Bayes': [],
    'KNN' : [],
    'SVM': [],
    'Perceptron': [],
    'Multi-Layer Perceptron': [],
    'Random Forest': []
}

global_dict = {
        'precision': [],
        'recall': [],
        'f1-score': []
}

gridsearch_params_dict = {
    'Decision Tree' : [],
    'Naive Bayes': [],
    'KNN' : [],
    'SVM': [],
    'Perceptron': [],
    'Multi-Layer Perceptron': [],
    'Random Forest': []
}

def get_average(y_true, y_pred):
    a1, a2, a3, a4 = score(y_true, y_pred, average='weighted')
    global_dict['precision'].append(a1) 
    global_dict['recall'].append(a2) 
    global_dict['f1-score'].append(a3)

def prepateToGraph(currentClassifier):
    global global_dict
    classifier_dict[currentClassifier].append(global_dict)
    
    global_dict = {
        'precision': [],
        'recall': [],
        'f1-score': []
    }

    

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def plot_results(classifierList):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    
    precision = []
    recall = []
    f1_score = []
    
    for classifier in classifierList:
        precision.append(np.mean(classifier_dict[classifier][0]['precision']))
        recall.append(np.mean(classifier_dict[classifier][0]['recall']))
        f1_score.append(np.mean(classifier_dict[classifier][0]['f1-score']))
    # data to plot
    n_groups = len(precision)
     
    # create plot
    
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8
     
    rects1 = plt.bar(index - bar_width, precision, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Precision')
     
    rects2 = plt.bar(index, recall, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Recall')
    
    rects3 = plt.bar(index + bar_width, f1_score, bar_width,
                     alpha=opacity,
                     color='r',
                     label='f1-score')
    
    plt.xlabel('Classifier Type')
    plt.ylabel('Scores')
    plt.title('Performance Comparison of Classifiers')
    plt.xticks(index, ('SVM', 'Decision Tree', 'KNN', 'Perceptron', 'MLP', 'Naive Bayes', 'Random Forest'))
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.legend()
    #TODO: 
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()  


def preprocess(filename):
    print("Pre-processing sensor data...")
    scam_data = pd.read_csv(filename, header = 0)
    # Feature Selection: All features are relevant.
    df = pd.DataFrame(scam_data)
    col_list = ["IRS Status","Tax related", "tax confidence", "arrest" , "prison", "privacy (Identity) threat",
                   "privacy (bank) threat","amount requested",
                   "payment methods", "scam signals", "court mentioned","urgency index"]
    # K-Best Features
    # See report to see why k = 130 was set.
    target = df[["scam"]]
    features = df[col_list]
    features = MultiColumnLabelEncoder(columns = col_list).fit_transform(features)
#    
#    selector = SelectKBest(chi2)
#    features  = selector.fit_transform(features, target)
#    idxs_selected = selector.get_support(indices=True)
#    # Create new dataframe with only desired columns, or overwrite existing
#    col_names = pd.DataFrame(df.columns[idxs_selected])
#    features = pd.DataFrame(features)
#    features.columns = col_names

    # normailise data
    features = normalize(features, norm='l2')
    
    dict = {'features':features, 'labels': target.values}
    return dict

def train_model(currentClassifier, filename):
    results = preprocess(filename)
    sss = StratifiedKFold(n_splits = 5, random_state = None, shuffle = False)
    features, labels = results['features'], results['labels'] 
    average_accuracy = list()
    classifierList = ['SVM', 'Decision Tree', 'KNN', 'Perceptron', 'Multi-Layer Perceptron', 'Naive Bayes', 'Random Forest']
    for classifier in classifierList:
        currentClassifier = classifier
        for train_index, test_index in sss.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            classifierDict = {"Decision Tree": [DecisionTreeClassifier(), {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': range(2, 20)}],
                               "KNN": [KNeighborsClassifier(), 
                                       {'n_neighbors': [1, 2 , 4 , 8], 'weights': ['uniform']}],
                               "Perceptron": [Perceptron(), {'alpha': [0, 1e-20, 1e-15, 1e-10, 1e-5, 1e-2]}],
                               "Multi-Layer Perceptron": [MLPClassifier(),{'solver':['lbfgs','sgd','adam'], 
                                                           'alpha':[1e-20, 1e-15, 1e-10, 1e-5, 1e-2], 
                                                           'hidden_layer_sizes': (2,10)}],
                               "SVM": [SVC(), [{'kernel': ['rbf'], 'gamma': [1/len(results['features'])], # 1/(number of features)
                                        'C': [0.125, 0.25, 0.5, 1, 2,4]},
                                        {'kernel': ['linear'], 'C': [0.125, 0.25, 0.5, 1, 2, 4]}]],
                              "Random Forest": [RandomForestClassifier(), {'max_depth': range(1,20)}]
            }
            
            print("Applying ", currentClassifier, " Please wait...")
            if currentClassifier == "Naive Bayes":
                clf = GaussianNB()
                clf.fit(X_train, y_train)
            else:
                gridsearch = GridSearchCV(classifierDict[currentClassifier][0], 
                                     classifierDict[currentClassifier][1], refit = 'AUC', cv = sss)
                clf =  make_pipeline(StandardScaler(),
                                     gridsearch)
                clf.fit(X_train, y_train) 
                gridsearch_params_dict[currentClassifier].append(gridsearch.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            print("Detailed classification report for :", currentClassifier)
            print()
            print("The model is trained on 80% of input data.")
            print("The scores are computed on the evaluation set (20% of input data).")
            print("Training complete")
            y_true, y_pred = y_test, clf.predict(X_test)
            print("Accuracy Scrore: ", currentClassifier, accuracy_score(y_true, y_pred))
            average_accuracy.append(accuracy_score(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            get_average(y_true, y_pred)
            print("Completed!")
        
        model_file = 'scam_' + currentClassifier + '.pkl'
        joblib.dump(clf, model_file) 
        print("Model saved to file", model_file)
        print(clf)
        print("Training average of ", currentClassifier, "is: ", np.mean(average_accuracy))
        prepateToGraph(currentClassifier)
    
    plot_results(classifierList)
    
def predict(currentClassifier, filename_predict):
    print("Making Predictions...")
    results = preprocessing(filename_predict, False)
    model_file = 'scam_' + currentClassifier + '.pkl'
    clf_predict = joblib.load(model_file) 
    y_true, y_pred = results['labels'], clf_predict.predict(results['features'])
    print("Accuracy Score: ", currentClassifier, accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("Completed!")
    
if __name__ == "__main__":
    classifier = "Naive Bayes"
    file = "call_data.csv"
    train_model(classifier, file)
    #predict(classifier, file)

