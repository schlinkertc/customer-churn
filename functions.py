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
        
def make_line(node_id,child_list,line=[]):
    if child_list[node_id] == -1:
        return line
    else:
        line.append(child_list[node_id])
        return make_line(child_list[node_id],child_list,line=line)

def plotly_decisionTree(estimator):
    tree = estimator.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    node_depth = np.zeros(shape=n_nodes,dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes,dtype=bool)
    stack = [(0,-1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node 
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    node_structure = {}
    node_heights = [np.abs(x-np.max(node_depth)) for x in node_depth]
    for node_id in range(n_nodes):
        node_dict = {
            'height':node_heights[node_id]+1
        }
        if node_id == 0: # if it's the first parent node
            node_dict['x_position'] = 0
        else:
            #find parent node
            if node_id in children_left:
                parent_id = children_left.tolist().index(node_id)
                parent_node = node_structure[parent_id]
            if node_id in children_right:
                parent_id = children_right.tolist().index(node_id)
                parent_node = node_structure[parent_id]
            
            # set x position of node based on parent node 
            if node_id == parent_node['children'][0]:
                node_dict['x_position'] = parent_node['x_position'] - np.sqrt((1/node_dict['height']))
            elif node_id == parent_node['children'][1]:
                node_dict['x_position'] = parent_node['x_position'] + np.sqrt((1/node_dict['height']))
            else:
                node_dict['x_position'] = parent_node['x_position']

        node_dict['children'] = (children_left[node_id],children_right[node_id])
        node_dict['children_left'] = children_left[node_id]
        node_dict['children_right'] = children_right[node_id]

        node_structure[node_id]=node_dict

    structure = pd.DataFrame(node_structure).T
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=structure['x_position'],
        y=structure['height'],
        mode='markers+text',
        text=structure.index
    ))
    for child_list in children_left,children_right:
        L = []
        for n in range(nodes):
            temp = make_line(n,child_list,line=[]) 
            if temp == []:
                continue 
            else:
                L.append(temp)

        for l in L:
            _line = [(node_structure[x]['x_position'],node_structure[x]['height']) for x in l]
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in _line],
                    y=[p[1] for p in _line],
                    mode = 'lines',
                )
            )
    return fig