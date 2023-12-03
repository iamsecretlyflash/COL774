from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from matplotlib import pyplot as plt
from DecisionTree import DecisionTree

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
types = [1 if x == 'cat' else 0 for x in types]

depths = [15, 25, 35, 45]
for max_depth in depths : 
    print(max_depth)
    tree = DecisionTree()
    tree.fit(X_train, y_train, types, max_depth=max_depth)
    tot_nodes = 0
    for i in list(tree.node_level_dict.values()):
        tot_nodes += len(i)
    val_acc, train_acc, test_acc, net_trimmed = tree.post_prune(X_val, y_val, X_train, y_train, X_test, y_test)
    plt.plot([tot_nodes -i for i in range(1,net_trimmed+1)], val_acc, label = f'Validation')
    plt.plot([tot_nodes -i for i in range(1,net_trimmed+1)], train_acc, label = f'Train')
    plt.plot([tot_nodes -i for i in range(1,net_trimmed+1)], test_acc, label = f'Test')
    plt.xlabel('Number of nodes ')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Number of nodes for max_depth = {max_depth}')
    plt.savefig(f'Q1/plots/part_c/part_c_{max_depth}.png')
    plt.clf()
    plt.close()