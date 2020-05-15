import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # data updates 
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})


    df['InternetType']=df['InternetService']
    df['InternetService'] = df['InternetService'].map({'Fiber optic':'Yes','DSL':'Yes','No':'No'})

    product_features = [
        'PhoneService', 'InternetService', 
        'OnlineBackup', 'OnlineSecurity', 
        'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies'
    ]

    df['Phone Lines'] = df['MultipleLines'].map({'No phone service':0,'No':1,'Yes':2})
    df.drop(columns=['MultipleLines'],inplace=True)

    cat_features = ['Contract','PaymentMethod','gender']

    for feature in cat_features:
        df = df.merge(pd.get_dummies(df[feature],drop_first=True),right_index=True,left_index=True)
        df.drop(columns=[feature],inplace=True)

    df = df.join(pd.get_dummies(df['InternetType']).drop(columns=['No']))
    df = df.drop(columns='InternetType')

    df.replace({'Yes':1,'No':0},inplace=True)
    df.replace({'No internet service':0},inplace=True)

    df['TotalCharges']=df['TotalCharges'].replace({' ':np.nan}).astype(float)
    
    return df

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image  
from pydotplus import graph_from_dot_data


def plot_tree(estimator,X_train,y_train):    
    # Create DOT data
    dot_data = export_graphviz(estimator, out_file=None, 
                               feature_names=X_train.columns,  
                               class_names=np.unique(y_train).astype('str'), 
                               filled=True, rounded=True, special_characters=True)

    # Draw graph
    graph = graph_from_dot_data(dot_data)  

    # Show graph
    return Image(graph.create_png())

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true,predicted,classes):
    import itertools
    cm=confusion_matrix(true,predicted,labels=classes)
    
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm,cmap=plt.cm.Blues)
    #plt.title('Confusion matrix',fontdict={'size':20})
    fig.colorbar(cax)
    ax.set_xticklabels([''] + classes,fontdict={'size':14})
    ax.set_yticklabels([''] + classes,fontdict={'size':14})
    plt.xlabel('Predicted',fontdict={'size':14})
    plt.ylabel('True',fontdict={'size':14})
    plt.grid(b=None)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             fontdict={'size':14,'weight':'heavy'},
             color="black" if cm[i, j] > thresh else "black")