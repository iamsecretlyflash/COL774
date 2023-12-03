from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import dataframe_image as dfi
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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

parameters = {'n_estimators': [i for i in range(50,450,100)],
              'max_features' :[0.1,0.3,0.5,0.7,0.9],
              'min_samples_split':[ i for i in range(2,11,2)]}


rf_model = RandomForestClassifier(oob_score=True)
best_score = float('-inf')
for g in ParameterGrid(parameters):
    rf_model.set_params(**g)
    rf_model.fit(X_train,y_train.T[0])
    # save if best
    if rf_model.oob_score_ > best_score:
        best_score = rf_model.oob_score_
        best_grid = g
        print(best_grid, best_score)

print( "Grid:", best_grid)
rf_model.set_params(**best_grid)
rf_model.fit(X_train,y_train.T[0])
print("The following values are obtained for the best set of paramaters:")
print("Training Accuracy:",rf_model.score(X_train,y_train.T[0]))
print("Test Accuracy:",rf_model.score(X_test,y_test.T[0]))
print("Validation Accuracy:",rf_model.score(X_val,y_val.T[0]))
print("OOB Score:",rf_model.oob_score_)
'''
Result Obtained:
Grid: {'max_features': 0.9, 'min_samples_split': 8, 'n_estimators': 250}
The following values are obtained for the best set of paramaters:
Training Accuracy: 0.9793024147182828
Test Accuracy: 0.7280248190279214
Validation Accuracy: 0.7091954022988506
OOB Score: 0.721732464545803'''