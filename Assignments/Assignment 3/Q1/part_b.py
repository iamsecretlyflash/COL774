from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import dataframe_image as dfi
from matplotlib import pyplot as plt
from DecisionTree import DecisionTree
import os

label_encoder = None 

def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()

X_train,y_train = get_np_array('train.csv')
X_test, y_test = get_np_array("test.csv")
types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]
while(len(types) != X_train.shape[1]):
        types = ['cat'] + types

types = [1 if x == 'cat' else 0 for x in types]
depth = [15, 25, 35, 45, 55, 75,100]
train_acc = []; test_acc = []
for max_depth in depth:
    print(f"Max Depth : {max_depth}")
    tree = DecisionTree()
    tree.fit(X_train,y_train,types, max_depth = max_depth)
    y_pred = tree(X_test)
    report = classification_report(y_test.T[0], y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(f"plots/part_b/test_{max_depth}.csv")
    dfi.export(report, f'plots/part_b/test_{max_depth}.png')
    test_acc.append(report.iloc[3][0])

    y_train_pred = tree(X_train)
    report = classification_report(y_train.T[0], y_train_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(f"plots/part_b/train_{max_depth}.csv")
    dfi.export(report, f'plots/part_b/train_{max_depth}.png')
    train_acc.append(report.iloc[3][0])

plt.figure(figsize=(12,8))
plt.plot(depth, train_acc, label = "Train Accuracy", color = 'orange')
plt.plot(depth, test_acc, label = "Test Accuracy", color = 'green')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs Max Depth")
plt.savefig("plots/part_b/accuracy.png")
    