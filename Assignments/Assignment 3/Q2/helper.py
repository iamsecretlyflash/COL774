import numpy as np 
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

def get_data(x_path, y_path):

    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalinete x:
    x = 2*(0.5 - x/255)
    return x, y

def get_metric(y_true, y_pred):
    
    results = classification_report(y_pred, y_true, output_dict=True)
    return results

def load():
    x_train_path = 'x_train.npy'
    y_train_path = 'y_train.npy'

    X_train, y_train = get_data(x_train_path, y_train_path)

    x_test_path = 'x_test.npy'
    y_test_path = 'y_test.npy'

    X_test, y_test = get_data(x_test_path, y_test_path)

    #you might need one hot encoded y in part a,b,c,d,e
    label_encoder = OneHotEncoder(sparse=False)
    label_encoder.fit(np.expand_dims(y_train, axis = -1))

    y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
    y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))

    return X_train, y_train_onehot, X_test, y_test_onehot