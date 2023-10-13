import cvxopt
import numpy as np
import os
from PIL import Image
import time
import pickle
from sklearn.svm import SVC as svm

C = 1

def resize(img) :
    img = np.array(img.resize((16,16)))
    return img.reshape(img.shape[0]*img.shape[1]*img.shape[2])/255

def loadClass(path):
    images = []
    for i in os.listdir(path):
        images.append(resize(Image.open(os.path.join(path,i))))
    return np.array(images)

def loadSVMData(c1,c2):
    class0 = loadClass(c1)
    label0 = (-1)*np.ones((class0.shape[0],1))
    class1 = loadClass(c2)
    label1 = np.ones((class1.shape[0],1))
    return np.concatenate([class0,class1]) ,np.concatenate([label0,label1])

def getSupport(arr,tol = 1e-3, C = 1):
    supportAlpha = []; indices = []
    for i in range(len(arr)):
        if arr[i] > tol :
            supportAlpha.append(arr[i])
            indices.append(i)
    return supportAlpha,indices

def linearKernel(X1,X2):
    return np.matmul(X1,X2.T)

def gaussKernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.001):
    prod = np.reshape(np.einsum('ij,ij->i', X1, X1), (X1.shape[0], 1)) + \
           np.reshape(np.einsum('ij,ij->i', X2, X2), (X2.shape[0], 1)).T \
             - 2 * np.matmul(X1, X2.T)
    return np.exp(-gamma * prod)

def SVM(X, Y, kernel = 'gaussian' , gamma = 0.1, C = 1, tol = 1e-4,showProg = False):
    if kernel == 'linear':
        kernelMatrix = linearKernel(X,X)
    elif kernel == 'gaussian': 
        kernelMatrix = gaussKernel(X,X,gamma = gamma)
    print("kernel computed")

    P = cvxopt.matrix((kernelMatrix * np.matmul(Y,Y.T)))
    q = cvxopt.matrix(-np.ones(X.shape[0]))
    c = 0
    G = cvxopt.matrix(np.concatenate([np.eye(X.shape[0]),(-1)*np.eye(X.shape[0])]))
    h = cvxopt.matrix(np.concatenate([C*np.ones((X.shape[0],1)),np.zeros((X.shape[0],1))]))
    A = cvxopt.matrix(Y.T ,tc = 'd')
    b = cvxopt.matrix(0.0)

    sol = cvxopt.solvers.qp(P, q, G, h, A, b, options={'show_progress': showProg})
    alphaRaw = np.array(sol['x'])
    supportAlpha, supportIndices = getSupport(alphaRaw,tol, C)
    supportAlpha = np.array(supportAlpha)
    ySupport = Y[supportIndices]; xSupport = X[supportIndices]

    w = np.sum(supportAlpha * ySupport * xSupport , axis = 0)

    wXt = np.sum(alphaRaw * Y * kernelMatrix,axis = 0)

    M = max(supportIndices, key=lambda i: -float("inf") if Y[i] == 1  or C - alphaRaw[i] <= tol else wXt[i])
    m = min(supportIndices, key=lambda i: float("inf") if Y[i] == -1  or C - alphaRaw[i] <= tol else wXt[i])
    intercept = -(wXt[M] + wXt[m]) / 2
    return w,intercept,alphaRaw, supportIndices

# loading data
# entry number : 2021MT10236. Required classes : 0 and 1
# assigning label -1 to class 0 and label 1 to class 1
c1 = 0; c2 = 1
trainX, trainY = loadSVMData('data/svm/train/'+str(c1),'data/svm/train/'+str(c2))
testX, testY = loadSVMData('data/svm/val/'+str(c1),'data/svm/val/'+str(c2))

########
#PART A#
########
print("Binary Classification")
print("Part A")
t = time.time()
wLinear, bLinear ,alphaRawLinear ,supportLinear = SVM(trainX,trainY, kernel = 'linear',showProg = False, tol=1e-3, C = 1.0)
print(f"Time taken to train linear kernel SVM using CVXOPT = {time.time() -t}")

nSV = len(supportLinear)
print(f'Number of support vectors = {nSV}')
print(f'Number of support vectors make up {nSV*100/trainX.shape[0] : .3f}% of the training set')
print(f'Intercept term : {bLinear}')
wLinear = wLinear.reshape((wLinear.shape[0],1))
pred = np.matmul(wLinear.T,testX.T) + bLinear
pred = np.where(pred[0] >= 0, 1, -1)
testYtemp = testY.reshape(testY.shape[0])
accLinear = np.where((testYtemp == pred) == True)[0].shape[0]/testYtemp.shape[0]
print(f'Validation set accuracy = {accLinear*100 : .3f}%')

predTrain = np.matmul(wLinear.T,trainX.T) + bLinear
predTrain = np.where(predTrain[0] >= 0, 1, -1)
trainYtemp = trainY.reshape(trainY.shape[0])
accLinearTrain = np.where((trainYtemp == predTrain) == True)[0].shape[0]/trainYtemp.shape[0]
print(f'Training set accuracy = {accLinearTrain*100 : .3f}%')

vecs = [(255*trainX[i]).reshape(16,16,3).astype(np.uint8) for i in np.argsort(alphaRawLinear[supportLinear].flatten())[:6]]
for i in range(6):
    img = Image.fromarray(vecs[i])
    #img.save('plots/supportVectorLinear'+str(i)+'.png')
wtemp = 255*wLinear.reshape((16,16,3)).astype(np.uint8)
img  = Image.fromarray(wtemp)
#img.save('plots/wLinear.png')

print("PART B: Gaussian Kernel")
#gaussian prediction
t = time.time()
_ , bGaussian ,alphaRawGaussian ,supportGaussian = SVM(trainX,trainY, kernel = 'gaussian',gamma = 0.001,showProg = False, tol=1e-4, C = 1.0)
print(f"Time taken to train gaussian kernel SVM using CVXOPT = {time.time() -t}")

nSVg = len(supportGaussian)
print(f'Number of support vectors = {nSVg}')
print(f'Number of support vectors make up {nSVg*100/trainX.shape[0] : .3f}% of the training set')
print(f'Intercept term : {bGaussian}')
pred = np.sum(alphaRawGaussian[supportGaussian] * trainY[supportGaussian] * gaussKernel(trainX[supportGaussian], testX, 0.001), 0) + bGaussian
pred = np.where(pred>=0, 1, -1)
accGauss = np.where((testYtemp == pred) == True)[0].shape[0]/testYtemp.shape[0]
print(f'Validation set accuracy = {accGauss*100 : .3f}%')

matches = 0
for i in supportGaussian:
    if i in supportLinear:
        matches+=1
print("Matching support vectors = ",matches)
print(f"Number of matching support vectors = {matches}")

print("Storing Images")
vecs = [(255*trainX[i]).reshape(16,16,3).astype(np.uint8) for i in np.argsort(alphaRawGaussian[supportLinear].flatten())[:6]]
for i in range(6):
    img = Image.fromarray(vecs[i])
    img.save('images/Q2/supportVectorGauss'+str(i)+'.png')

########
#PART C#
########


linSVM = svm(kernel='linear', C = C)
t = time.time()
linSVM.fit(trainX,trainY.flatten())
print(f'Time taken to train Sklearn Linear SVM = {time.time() - t : .3f}s')


gaussSVM = svm(kernel='rbf', C = C, gamma = 0.001)
t = time.time()
gaussSVM.fit(trainX,trainY.flatten())
print(f'Time taken to train Sklearn Gaussian SVM = {time.time() - t : .3f}s')

print(f'Using Sci-Kit Learn\nnSV for linear = {linSVM.support_vectors_.shape[0]}\nnSV for Gaussian = {gaussSVM.support_vectors_.shape[0]}')
matches = 0
for i in gaussSVM.support_:
    if i in linSVM.support_:
        matches+=1
print("Matching support vectors = ",matches)
print(f'bias for linear :{linSVM.intercept_[0]: .4f}')
print(f'norm of difference between wLinear from cvxopt and sklearn : {np.linalg.norm(wLinear.flatten() - linSVM.coef_) : .4f}')
predLinearSK = linSVM.predict(testX)
predLinearSK = np.where(predLinearSK>=0, 1, -1)
accLinearSK = np.where((testY.flatten() == predLinearSK) == True)[0].shape[0]/testY.shape[0]
print(f'Accuracy on validation set using sklearn Linear SVM = {accLinearSK}')
predGaussSK = gaussSVM.predict(testX)
predGaussSK = np.where(predGaussSK>=0, 1, -1)
accGaussSK = np.where((testY.flatten() == predGaussSK) == True)[0].shape[0]/testY.shape[0]
print(f'Accuracy on validation set using sklearn Gaussian SVM = {accGaussSK}')

print("PART -D  : EXTRA FUN!!")
def resizeD(img) :
    img = np.array(img.resize((32,32)))
    return img.reshape(img.shape[0]*img.shape[1]*img.shape[2])/255

def loadClassD(path):
    images = []
    for i in os.listdir(path):
        images.append(resizeD(Image.open(os.path.join(path,i))))
    return np.array(images)

def loadSVMDataD(c1,c2):
    class0 = loadClassD(c1)
    label0 = (-1)*np.ones((class0.shape[0],1))
    class1 = loadClassD(c2)
    label1 = np.ones((class1.shape[0],1))
    return np.concatenate([class0,class1]) ,np.concatenate([label0,label1])

# loading data
# entry number : 2021MT10236. Required classes : 0 and 1
# assigning label -1 to class 0 and label 1 to class 1
c1 = 0; c2 = 1
trainX, trainY = loadSVMDataD('data/svm/train/'+str(c1),'data/svm/train/'+str(c2))
testX, testY = loadSVMDataD('data/svm/val/'+str(c1),'data/svm/val/'+str(c2))
import time
linSVM = svm(kernel='linear', C = C)
t = time.time()
linSVM.fit(trainX,trainY.flatten())
print(f'Time taken to train Sklearn Linear SVM = {time.time() - t : .3f}s')


gaussSVM = svm(kernel='rbf', C = C, gamma = 0.001)
t = time.time()
gaussSVM.fit(trainX,trainY.flatten())
print(f'Time taken to train Sklearn Gaussian SVM = {time.time() - t : .3f}s')

print(f'Using Sci-Kit Learn\nnSV for linear = {linSVM.support_vectors_.shape[0]}\nnSV for Gaussian = {gaussSVM.support_vectors_.shape[0]}')
print(f'bias for linear :{linSVM.intercept_[0]: .4f}')
predLinearSK = linSVM.predict(testX)
accLinearSK = np.where((testY.flatten() == predLinearSK) == True)[0].shape[0]/testX.shape[0]
print(f'Accuracy on validation set using sklearn Linear SVM = {accLinearSK}')
predGaussSK = gaussSVM.predict(testX)
accGaussSK = np.where((testY.flatten() == predGaussSK) == True)[0].shape[0]/testX.shape[0]
print(f'Accuracy on validation set using sklearn Gaussian SVM = {accGaussSK}')

########################################################################################################################

print("MULTI-CLASS CLASSIFICATION")
print("Part - A")

TRAIN_OR_LOAD = 1 # 0 for loading, 1 for training
# TRAINING ALL SVM MODELS
# to store : alphaRawGaussian, bGaussian and supportGaussian. Load trainX and trainY at time of prediction
CLASSES = 6
savePath = 'SVM_modelsQ2'
loadPath = 'SVM_modelsQ2/allaCombined.pickle'
modelsDict = {}
netTime = 0
if TRAIN_OR_LOAD == 1:
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    for i in range(CLASSES):
        for j in range(i+1,CLASSES):
            print(f'Training for class-{i} and class-{j}')
            trainX, trainY = loadSVMData('data/svm/train/'+str(i),'data/svm/train/'+str(j)) 
            tLocal = time.time()
            _ , bGaussian ,alphaRawGaussian ,supportGaussian = SVM(trainX,trainY, kernel = 'gaussian',gamma = 0.001,showProg = False, tol=1e-3, C = 1.0)
            netTime += time.time() - tLocal
            print(f'Time taken for {i} and {j} = {time.time() - tLocal : .3f}')

            #CHECKPOINTS
            with open(os.path.join(savePath,f'model{i}{j}.pickle'),'wb') as f:
                pickle.dump({'intercept' : bGaussian, 'alphaRaw' : alphaRawGaussian, 'supportIndices' : supportGaussian}, f)

            modelsDict[(i,j)] = {'intercept' : bGaussian, 'alphaRaw' : alphaRawGaussian, 'supportIndices' : supportGaussian}
    print(f'time taken to train cvxopt model = {netTime}')
    with open(os.path.join(savePath,'allaCombined.pickle'),'wb') as f:
        pickle.dump(modelsDict,f)
    temp = os.path.join(savePath,'allaCombined.pickle')
    print(f'Model saved : {temp}')
else:
    with open(loadPath,'rb') as f:
        modelsDict = pickle.load(f)

CLASSES = 6
for i in range(CLASSES):
    for j in range(i+1,CLASSES):
        trainX, trainY = loadSVMData('data/svm/train/'+str(i),'data/svm/train/'+str(j))
        modelsDict[(i,j)]['trainX'] = trainX
        modelsDict[(i,j)]['trainY'] = trainY

def getVotes(X,modelsDict):
    votes = {i:0 for i in range(CLASSES)}
    max_score = {i:0 for i in range(CLASSES)}
    X = X.reshape((1,X.shape[0]))
    for i in range(CLASSES):
        for j in range(i+1,CLASSES):
            pred = np.sum(modelsDict[(i,j)]['alphaRaw'][modelsDict[(i,j)]['supportIndices']] * modelsDict[(i,j)]['trainY'][modelsDict[(i,j)]['supportIndices']] \
                       * gaussKernel(modelsDict[(i,j)]['trainX'][modelsDict[(i,j)]['supportIndices']], X, 0.001), 0) + modelsDict[(i,j)]['intercept']
            predNum = np.where(pred <= 0 , i, j)[0]
            max_score[predNum] = max(max_score[predNum],pred[0])
            votes[predNum]+=1
    votes = {j:i for i,j in votes.items()}
    votes =  sorted(list(votes.items()))
    result = -1
    if votes[-1][0] == votes[-2][0]:
        if max_score[votes[-1][-1]] > max_score[votes[-2][-1]]:
            result = votes[-1][-1]
        else:
            result = votes[-2][1]
    else:
        result = votes[-1][1]
    return result

def loadAllClasses(tv = 'val'):

    class0 = loadClass('data/svm/'+tv+'/0')
    label0 = 0*np.ones((class0.shape[0],1))
    for i in range(1,6):
        class1 = loadClass('data/svm/'+ tv+ '/' + str(i))
        label1 = i*np.ones((class1.shape[0],1))
        class0, label0 = np.concatenate([class0,class1]) ,np.concatenate([label0,label1])
    return class0, label0

testX, testY = loadAllClasses()
trainX, trainY = loadAllClasses('train')

matches = 0
predsCustomSVM = []
for i in range(testX.shape[0]):
    predsCustomSVM.append(getVotes(testX[i],modelsDict))
    if predsCustomSVM[-1] == testY[i][-1] : matches += 1

print(f'Accuracy on validation = {matches/testX.shape[0]}')

print("PART B")
gaussSVM = svm( kernel='rbf', gamma = 0.001, C = 1)
t = time.time()
gaussSVM.fit(trainX, trainY.flatten())
print(f'time taken to train sklearn model = {time.time() - t}')
print(f'Accuracy on validation = {np.where(gaussSVM.predict(testX) == testY.flatten())[0].shape[0]/testY.shape[0]}')

print("Part - C")
from sklearn.metrics import confusion_matrix as cm
import pandas as pd
matrix = cm(testY.flatten(),predsCustomSVM)
cm_df = pd.DataFrame(matrix,dtype=np.int32)
print("The confusion matrix for the given task using custom SVM : ")
print(cm_df)
vecs =  [np.random.choice(np.where((np.array(predsCustomSVM) != testY.flatten()) == True)[0], 12)]
vec = [(255*trainX[i]).reshape(16,16,3).astype(np.uint8) for i in vecs[0]]
for i in range(12):
    img = Image.fromarray(vec[i])
    #img.save('plots/misclassified'+str(i)+'.png')

matrixsk = cm(list(testY.flatten()),list(gaussSVM.predict(testX)))
cm_dfsk = pd.DataFrame(matrixsk,dtype=np.int32)
print("The confusion matrix for the task using sklearn : ")
cm_dfsk

print("PART - D")
gamma = 0.001
checkC = [1e-5,1e-3,1,5,10]
trainCombined = np.concatenate([trainX, trainY], axis = 1)
np.random.shuffle(trainCombined)
trainX, trainY = trainCombined[:,:-1], trainCombined[:,-1:]
accDict = {}
K = 5
valSize = trainX.shape[0]//K
for c in checkC :
    bestAcc = 0; bestModel = None; avgAcc = 0
    for i in range(K):
        svmModel = svm(kernel='rbf', gamma=gamma, C = c)
        ind = np.where(np.logical_or(np.array([i for i in range(trainX.shape[0])])>=(i+1)*valSize, np.array([i for i in range(trainX.shape[0])])<i*valSize) == True)[0]
        svmModel.fit(trainX[ind], trainY[ind].flatten())
        preds = svmModel.predict(trainX[i*valSize:(i+1)*valSize])
        acc = np.where((trainY[i*valSize:(i+1)*valSize].flatten() == preds) == True)[0].shape[0]/valSize
        if acc > bestAcc :
            bestAcc = acc
            bestModel = svmModel
        avgAcc += acc
    #validation accuracy
    svmModel = svm(kernel='rbf', gamma=gamma, C = c)
    svmModel.fit(trainX, trainY.flatten())
    preds = svmModel.predict(testX)
    valAcc = np.where((testY.flatten() == preds) == True)[0].shape[0]/testY.shape[0]
    accDict[c] = (avgAcc/K, valAcc, bestModel)
import pickle
with open('SVM_modelsQ2/crossFold.pickl','wb') as f:
    pickle.dump(accDict,f)
accDict
from matplotlib import pyplot as plt
plt.figure(figsize=(12,10))
plt.plot(np.log10(checkC),[accDict[i][0] for i in accDict] ,label = 'crossFoldAcc')
plt.plot(np.log10(checkC),[accDict[i][1] for i in accDict] , label = 'valAcc')
plt.legend()
plt.title('CrossFoldAcc and Val Acc vs log(C)')
#plt.savefig('plots/CrossFoldGraph.png')