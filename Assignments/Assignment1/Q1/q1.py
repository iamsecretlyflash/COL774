#Linear Regression
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

def gradient(X,Y,theta):
    #print(f'Shapes of inputs : \nX: {X.shape}\nY: {Y.shape}\ntheta: {theta.shape})
    return ((np.matmul(X.T,(np.matmul(X,theta)-Y))))/(X.shape[0])  #dividing by batch size for exrta numerical stability

def loss(X,Y,theta):
    #print(f'Shapes of inputs : \nX: {X.shape}\nY: {Y.shape}\ntheta: {theta.shape})
    return np.square(np.linalg.norm(Y -np.matmul(X,theta)))/(2*X.shape[0])

def gradDescent(X,Y,batch_size = 1, lr = 0.05, stopThresh = 1e-15):
    #print(f'Shapes of inputs : \nX: {X.shape}\nY: {Y.shape})
    print(f'Learning rate(eta) = {lr}\nStopping Threshold = {stopThresh}')

    theta = np.zeros(X.shape[1]).reshape(X.shape[1],1)

    curr_loss = 1; prev_loss = 0
    thetaMov = [(theta[0][0],theta[1][0],loss(X,Y,theta))]
    epochs = 0

    t = time.time()
    while abs(curr_loss-prev_loss)>stopThresh:
        net_loss = 0
        theta = theta - lr*gradient(X,Y,theta)
        lossVal =  loss(X,Y,theta)
        thetaMov.append((theta[0][0],theta[1][0],lossVal))
        prev_loss = curr_loss
        curr_loss = lossVal
        epochs += 1

    print(f'Training complete in: {time.time()-t}')
    print(f'Epochs taken: {epochs}')
    print(theta)

    return theta,np.array(thetaMov)

def loadData(path):
    return np.loadtxt(path)

def normalizeData(x):
    return (x-np.mean(x,axis = 0))/np.std(x,axis = 0)

trainX = normalizeData(loadData('data/q1/linearX.csv'))
trainY = (loadData('data/q1/linearY.csv'))
trainX = trainX.reshape(len(trainX),1)
trainY = trainY.reshape(len(trainY),1)
trainX = np.concatenate([np.ones(trainX.shape[0]).reshape(trainX.shape[0],1),trainX],axis = 1)

theta, thetaMovement = gradDescent(trainX,trainY,lr = 0.01)

plt.figure(figsize=(15,9))
ax = plt.axes()
ax.set_facecolor('white')
plt.scatter(trainX[:,1],trainY,s = 15,label = "Training Data")
plt.plot(trainX[:,1],np.matmul(trainX,theta).reshape(trainX.shape[0],1),linewidth = 2,label = "Learned Hypothesis",c = 'orange')
plt.legend(loc = "lower right")
plt.xlabel('Wine Acidity')
plt.ylabel('Wine Density')
plt.title("Linear Regression over wine acidity")
plt.savefig('images/Q1linearRegressionHypothesis.png')
plt.show()

#plotting 
t1,t2 = np.meshgrid(np.linspace(-0.2,2,200),np.linspace(-1.2,1.2,200))
lossFuncVals = np.apply_along_axis(
        lambda theta: loss( trainX,trainY,np.reshape(theta, (-1, 1))),
        2, np.stack([t1, t2], axis=-1))

#plotting the 3D loss function
theta0, theta1 = t1,t2
fig = plt.figure(figsize=(15,9))
axes = fig.add_subplot(title='Movement of theta with iterations', projection='3d', xlabel='theta0', ylabel='theta1', zlabel='Loss')
surface = axes.plot_surface(theta0, theta1, lossFuncVals, label='Loss Function', alpha=0.5,cmap = cm.gist_rainbow)
surface._facecolors2d = surface._facecolor3d
surface._edgecolors2d = surface._edgecolor3d

#animation part
learn, = axes.plot([], [], [], color='red', label='Change in Theta')
thetaMovement = thetaMovement.T
axes.legend()
def update(iteration):
        learn.set_data(thetaMovement[:2, :iteration+1])
        learn.set_3d_properties(thetaMovement[2, :iteration+1])
        return learn,

anim = animation.FuncAnimation(fig, update,
                                   frames=range(thetaMovement.shape[1]),
                                   interval=200, blit=True)
update(309)
plt.show()
fig.savefig('images/thetaMovement.png')
plt.close()
print('3D Loss Function ANIMATION DONE')

#2D contour animation

fig = plt.figure(figsize=(10,6))
axes = fig.add_subplot(title ='Contours for Learning rate(eta) = ' + str(0.05), xlabel = 'theta0',ylabel = 'theta1' )
axes.contour(theta0, theta1,lossFuncVals , 100,label = 'Cost function contours')
learn, = axes.plot([], [], color='red', marker='x',
                         linestyle='None', label='Change in Theta')
axes.legend()
def update(iteration):
        learn.set_data(thetaMovement[:2, :iteration+1])
        return learn,

anim = animation.FuncAnimation(fig, update,
                                   frames=range(thetaMovement.shape[1]),
                                   interval=200, blit=True)

plt.show()
#fig.savefig('images/q1d.png')
plt.close()
print("2nd animation done")

#for part e
etas = [0.001,0.025,0.1]
i = 0
for lr in etas:
    print(lr)
    theta, thetaMovement = gradDescent(trainX,trainY, lr = lr)
    print(theta)
    pltTitle = 'Contours for Learning rate(eta) = ' + str(lr)
    if lr == 0.001:
        print(f'LR = {lr}. Skipping some thetas in middle')
        pltTitle = f'Contours for Learning rate(eta) = {str(lr)} (Skipping thetas in middle)' 
        thetaMovement = thetaMovement[0:-1:20]
    thetaMovement = thetaMovement.T
    fig = plt.figure(figsize=(15,9))
    axes = fig.add_subplot(title =pltTitle, xlabel = 'theta0',ylabel = 'theta1' )
    axes.contour(theta0, theta1,lossFuncVals , 100,label = 'Cost function contours')
    learn, = axes.plot([], [], color='red', marker='x',
                            linestyle='None', label='Change in Theta')
    axes.legend()
    def update(iteration):
            learn.set_data(thetaMovement[:2, :iteration+1])
            return learn,

    anim = animation.FuncAnimation(fig, update,
                                    frames=range(thetaMovement.shape[1]),
                                    interval=200, blit=True)
    i+=1
    plt.show()
    #fig.savefig(f'images/q1e{i}.png')
    plt.close()