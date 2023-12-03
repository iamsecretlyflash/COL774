from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
from sklearn.metrics import classification_report
import dataframe_image as dfi
from matplotlib import pyplot as plt
from DecisionTree import DecisionTree
import os
print(os.listdir())

label_encoder = None 

def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OrdinalEncoder()
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()

#change the path if you want
X_train,y_train = get_np_array('Q1/train.csv')
X_test, y_test = get_np_array("Q1/test.csv")

types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]
types = [1 if i == 'cat' else 0 for i in types]

depth = [5, 10, 15, 20, 25]
train_acc = []; test_acc = []
for max_depth in depth:
    print(f"Max Depth : {max_depth}")
    tree = DecisionTree()
    tree.fit(X_train,y_train,types, max_depth = max_depth)
    y_pred = tree(X_test)
    report = classification_report(y_test.T[0], y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(f"Q1/plots/test_{max_depth}.csv")
    dfi.export(report, f'Q1/plots/test_{max_depth}.png')
    test_acc.append(report.iloc[3][0])

    y_train_pred = tree(X_train)
    report = classification_report(y_train.T[0], y_train_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(f"Q1/plots/train_{max_depth}.csv")
    dfi.export(report, f'Q1/plots/train_{max_depth}.png')
    train_acc.append(report.iloc[3][0])
    
depth = [5, 10, 15, 20, 25]
plt.figure(figsize=(12,8))
plt.plot(depth, train_acc, label = "Train Accuracy", color = 'orange')
plt.plot(depth, test_acc, label = "Test Accuracy", color = 'green')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs Max Depth")
plt.savefig("Q1/plots/accuracy.png")