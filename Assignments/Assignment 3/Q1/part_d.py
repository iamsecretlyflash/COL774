from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import dataframe_image as dfi
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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

X_train,y_train = get_np_array('Q1/train.csv')
X_test, y_test = get_np_array("Q1/test.csv")
X_val, y_val = get_np_array("Q1/val.csv")
types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]
while(len(types) != X_train.shape[1]):
    types = ['cat'] + types

depths = [15, 25, 35, 45]
train_acc = []; test_acc = []
for max_depth in depths:
    print(max_depth)
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = (classification_report(y_test.T[0], y_pred, output_dict=True))
    report = pd.DataFrame(report).transpose()
    report.to_csv(f"Q1/plots/part_d/test_{max_depth}.csv")
    dfi.export(report, f'Q1/plots/part_d/test_{max_depth}.png')
    test_acc.append(report.iloc[3][0])
    print("Test report\n:",report)
    y_train_pred = clf.predict(X_train)
    report = classification_report(y_train.T[0], y_train_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv(f"Q1/plots/part_d/train_{max_depth}.csv")
    dfi.export(report, f'Q1/plots/part_d/train_{max_depth}.png')
    train_acc.append(report.iloc[3][0])
    print("------------------------------------------------------------")
    print()
    
plt.figure(figsize=(12,8))
plt.plot(depths, train_acc, label = "Train Accuracy", color = 'orange')
plt.plot(depths, test_acc, label = "Test Accuracy", color = 'green')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs Max Depth")
plt.savefig("Q1/plots/part_d/accuracy.png")

#Using validations set 
best_acc = float('-inf')
best_depth = None
acc_list = []
for max_depth in [15,25,35,45]:

    clf = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy')
    clf.fit(X_val, y_val)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test.T[0])
    print(max_depth, acc)
    acc_list.append(acc)
    if(acc > best_acc):
        best_acc = acc
        best_depth = max_depth
plt.clf()
plt.plot([15,25,35,45], acc_list, label = "Accuracy", color = 'orange')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs Max Depth")
plt.savefig("Q1/plots/part_d/depth_accuracy_val.png")

# ccp_alpha
ccp_alphas = [0.001, 0.01, 0.1, 0.2]
best_acc = float('-inf')
best_ccp = None
cpp_list = []
for ccp_alpha in ccp_alphas:

    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha, criterion='entropy')
    clf.fit(X_val, y_val)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test.T[0])
    cpp_list.append(acc)
    print(ccp_alpha, acc)
    if(acc > best_acc):
        best_acc = acc
        best_ccp = ccp_alpha
plt.clf()
plt.plot(ccp_alphas, cpp_list, label = "Accuracy", color = 'orange')
plt.xlabel("CCP Alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs CCP Alpha")
plt.savefig("Q1/plots/part_d/ccp_accuracy_val.png")