import numpy as np 
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from NeuralNetwork import NeuralNetwork as NN
from helper import load, get_metric
import pickle
import pandas as pd
import dataframe_image as dfi
import os

if os.path.exists("plots") == False:
    os.mkdir("plots")
if os.path.exists("plots/part_a") == False:
    os.mkdir("plots/part_a")
if os.path.exists("plots/part_b") == False:
    os.mkdir("plots/part_b")
if os.path.exists("plots/part_c") == False:
    os.mkdir("plots/part_c")
if os.path.exists("plots/part_d") == False:
    os.mkdir("plots/part_d")
if os.path.exists("plots/part_e") == False:
    os.mkdir("plots/part_e")
if os.path.exists("plots/part_f") == False:
    os.mkdir("plots/part_f")
if os.path.exists("models") == False:
    os.mkdir("models")

trainX, trainY, testX, testY = load()

#part B
layers_to_test = [[1024, 1 , 5], [1024, 5 , 5], [1024, 10 , 5] , [1024, 50 , 5], [1024, 100 , 5]]
for layer_info in layers_to_test:
    print("Training architecture : ", layer_info)
    network1 = NN(layer_info, activation_func='sigmoid')
    network1.fit(trainX, trainY, lambda x : 0.01, 1000, batch_size=32, print_every= 200, stop_thresh=1e-6)
    network1DICT = {"W": network1.W, "b" : network1.b, "epoch_losses" : network1.epoch_losses, "hidden_layers" : network1.layers_config , "activation" : network1.activation_function, "leakyReluSlope" : network1.leakyRelu_slope}
    model_name = "models/part_b/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.pickle"
    with open(model_name,'wb') as f:
        pickle.dump(network1DICT,f)
    report = get_metric(testY.argmax(axis = 1), network1.predict(testX.T))
    report = pd.DataFrame(report).transpose()
    result = "plots/part_b/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.csv"
    report.to_csv(result)

from matplotlib import pyplot as plt
average_acc = []
for layer_info in layers_to_test:
    result = "plots/part_b/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.csv"
    report = pd.read_csv(result)
    average_acc.append(report.iloc[5][1])
plt.plot([1,5,10,50,100], average_acc)
plt.xlabel("Number of hidden layers")
plt.ylabel("Average accuracy")
plt.title("Average accuracy vs Number of hidden layers")
plt.savefig("plots/part_b/average_accuracy_vs_hidden_layers.png")

# PART C

layers_to_test = [[1024, 512 , 5], [1024, 512, 256 , 5], [1024, 512, 256, 128 , 5] , [1024, 512, 256, 128, 64 , 5]]
for layer_info in layers_to_test:
    print("Training architecture : ", layer_info)
    network1 = NN(layer_info, activation_func='sigmoid')
    network1.fit(trainX, trainY, lambda x : 0.01, 1000, batch_size=32, print_every= 200, stop_thresh=1e-6)
    network1DICT = {"W": network1.W, "b" : network1.b, "epoch_losses" : network1.epoch_losses, "hidden_layers" : network1.layers_config , "activation" : network1.activation_function, "leakyReluSlope" : network1.leakyRelu_slope}
    model_name = "models/part_c/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.pickle"
    with open(model_name,'wb') as f:
        pickle.dump(network1DICT,f)
    report = get_metric(testY.argmax(axis = 1), network1.predict(testX.T))
    report = pd.DataFrame(report).transpose()
    result = "plots/part_c/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.csv"
    report.to_csv(result)

average_acc = []
for layer_info in layers_to_test:
    result = "plots/part_c/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.csv"
    report = pd.read_csv(result)
    average_acc.append(report.iloc[5][1])
plt.plot([1,2,3,4], average_acc)
plt.xlabel("Number of hidden layers")
plt.ylabel("Average accuracy")
plt.savefig("plots/part_c/average_accuracy_vs_hidden_layers.png")

import dataframe_image as dfi
for layer_info in layers_to_test:
    result = "plots/part_c/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid"
    report = pd.read_csv(result + '.csv')
    dfi.export(report,result + ".png")

# train metrics
for layer_info in layers_to_test:
    model_name = "models/part_c/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.pickle"
    with open(model_name,'rb') as f:
        network1DICT = pickle.load(f)
    network1 = NN(layer_info, activation_func='sigmoid')
    network1.W = network1DICT["W"]
    network1.b = network1DICT["b"]
    network1.layers_config = network1DICT["hidden_layers"]
    network1.activation_function = network1DICT["activation"]
    network1.leakyRelu_slope = network1DICT["leakyReluSlope"]
    report = get_metric(trainY.argmax(axis = 1), network1.predict(trainX.T))
    report = pd.DataFrame(report).transpose()
    result = "plots/part_c/train_model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid"
    report.to_csv(result + ".csv")
    dfi.export(report,result + ".png")

average_acc = []
for layer_info in layers_to_test:
    result = "plots/part_c/train_model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.csv"
    report = pd.read_csv(result)
    average_acc.append(report.iloc[5][1])
plt.plot([1,2,3,4], average_acc)
plt.xlabel("Number of hidden layers")
plt.ylabel("Average accuracy")
plt.savefig("plots/part_c/train_average_accuracy_vs_hidden_layers.png")

# PART D

layers_to_test = [[1024, 512 , 5], [1024, 512, 256 , 5], [1024, 512, 256, 128 , 5] , [1024, 512, 256, 128, 64 , 5]]
for layer_info in layers_to_test:
    print("Training architecture : ", layer_info)
    network1 = NN(layer_info, activation_func='sigmoid')
    network1.fit(trainX, trainY, lambda x : 0.01/np.sqrt(x), 1000, batch_size=32, print_every= 200, stop_thresh=1e-6)
    network1DICT = {"W": network1.W, "b" : network1.b, "epoch_losses" : network1.epoch_losses, "hidden_layers" : network1.layers_config , "activation" : network1.activation_function, "leakyReluSlope" : network1.leakyRelu_slope}
    model_name = "models/partd_d/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.pickle"
    with open(model_name,'wb') as f:
        pickle.dump(network1DICT,f)
    report = get_metric(testY.argmax(axis = 1), network1.predict(testX.T))
    report = pd.DataFrame(report).transpose()
    result = "plots/part_d/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid"
    report.to_csv(result + '.csv')
    dfi.export(report,result + '.png')
    
    report_train = get_metric(trainY.argmax(axis = 1), network1.predict(trainX.T))
    report_train = pd.DataFrame(report_train).transpose()
    result_train = "plots/part_d/train_model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid"
    report_train.to_csv(result_train + ".csv")
    dfi.export(report_train,result_train + ".png")

network1DICT = {"W": network1.W, "b" : network1.b, "epoch_losses" : network1.epoch_losses, "hidden_layers" : network1.layers_config , "activation" : network1.activation_function, "leakyReluSlope" : network1.leakyRelu_slope}
model_name = "models/partd_d/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid.pickle"
with open(model_name,'wb') as f:
        pickle.dump(network1DICT,f)
report = get_metric(testY.argmax(axis = 1), network1.predict(testX.T))
report = pd.DataFrame(report).transpose()
result = "plots/part_d/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid"
report.to_csv(result + '.csv')
dfi.export(report,result + '.png')
    
report_train = get_metric(trainY.argmax(axis = 1), network1.predict(trainX.T))
report_train = pd.DataFrame(report_train).transpose()
result_train = "plots/part_d/train_model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_sigmoid"
report_train.to_csv(result_train + ".csv")
dfi.export(report_train,result_train + ".png")


#PART E

layers_to_test = [[1024, 512, 256 , 5], [1024, 512, 256, 128 , 5] , [1024, 512, 256, 128, 64 , 5]]
for layer_info in layers_to_test:
    print("Training architecture : ", layer_info)
    network1 = NN(layer_info, activation_func='ReLU')
    network1.fit(trainX, trainY, lambda x : 0.01 / np.sqrt(x), 1000, batch_size=32, print_every= 200, stop_thresh=1e-6)
    network1DICT = {"W": network1.W, "b" : network1.b, "epoch_losses" : network1.epoch_losses, "hidden_layers" : network1.layers_config , "activation" : network1.activation_function, "leakyReluSlope" : network1.leakyRelu_slope}
    model_name = "models/part_e/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_relu.pickle"
    with open(model_name,'wb') as f:
        pickle.dump(network1DICT,f)
    report_test = get_metric(testY.argmax(axis = 1), network1.predict(testX.T))
    report_test = pd.DataFrame(report_test).transpose()
    result_test = "plots/part_e/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_relu"
    report_test.to_csv(result_test + ".csv")
    dfi.export(report_test,result_test + ".png")

    report_train = get_metric(trainY.argmax(axis = 1), network1.predict(trainX.T))
    report_train = pd.DataFrame(report_train).transpose()
    result_train = "plots/part_e/train_model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_relu"
    report_train.to_csv(result_train + ".csv")
    dfi.export(report_train,result_train + ".png")
    
average_acc = []
for layer_info in layers_to_test:
    result = "plots/part_e/train_model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_relu.csv"
    report = pd.read_csv(result)
    average_acc.append(report.iloc[5][1])
plt.plot([1,2,3,4], average_acc)
plt.xlabel("Number of hidden layers")
plt.ylabel("Average accuracy")
plt.savefig("plots/part_e/train_average_accuracy_vs_hidden_layers.png")

average_acc = []
for layer_info in layers_to_test:
    result = "plots/part_e/model_H_" + "_".join([str(i) for i in layer_info]) + "_EPOCH_" + str(1000)+"_BATCH_32_activation_relu.csv"
    report = pd.read_csv(result)
    average_acc.append(report.iloc[5][1])
plt.plot([1,2,3,4], average_acc)
plt.xlabel("Number of hidden layers")
plt.ylabel("Average accuracy")
plt.savefig("plots/part_e/average_accuracy_vs_hidden_layers.png")

# PART F

layers_to_test = [[1024, 512 , 5], [1024, 512, 256 , 5], [1024, 512, 256, 128 , 5] , [1024, 512, 256, 128, 64 , 5]]
layers_to_test_MLP = [i[1:-1] for i in layers_to_test]
import time

for layer_info in layers_to_test_MLP:
    print("Testing layer info : ", layer_info)
    model = MLPClassifier(hidden_layer_sizes=layer_info, activation='relu', solver='sgd', max_iter=1000, learning_rate='invscaling', random_state=1)
    t = time.time()
    model.fit(trainX, trainY.argmax(axis = 1))
    print(f'Time taken to train MLP with {layer_info} hidden layers : {time.time() - t} seconds')
    report_test = get_metric(testY.argmax(axis = 1), model.predict(testX))
    report_test = pd.DataFrame(report_test).transpose()
    result_test = "plots/part_f/model_H_" + "_".join([str(i) for i in layer_info]) + "_BATCH_EPOCH_1000_32_activation_relu"
    report_test.to_csv(result_test + ".csv")
    dfi.export(report_test,result_test + ".png")

    report_train = get_metric(trainY.argmax(axis = 1), model.predict(trainX))
    report_train = pd.DataFrame(report_train).transpose()
    result_train = "plots/part_f/train_model_H_" + "_".join([str(i) for i in layer_info]) + "_BATCH_EPOCH_1000_32_activation_relu"
    report_train.to_csv(result_train + ".csv")
    dfi.export(report_train,result_train + ".png")