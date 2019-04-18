import scipy.io as spio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#read data
def readData() :
    data = spio.loadmat('mnistReduced.mat')

    xTrain = np.array(data['images_train'])  # 784*30000
    yTrain = np.array(data['labels_train'])  # 1*30000

    xVal = np.array(data['images_val'])  # 784*3000
    yVal = np.array(data['labels_val']) # 1*3000

    xTest = np.array(data['images_test']) # 784*3000
    yTest = np.array(data['labels_test']) # 1*3000

    return np.transpose(xTrain), yTrain, np.transpose(xVal), yVal, np.transpose(xTest), yTest

#Check for data imbalance
def dataBalanced(labels) :
    total = np.zeros((10,1))
    val = 0
    for i in labels :
        for j in i:
            total[j] += 1
    for i in total :
        for j in i :
            val += j
    print("distribution of class labels:")
    for i in range(len(total)) :
        print(str(i) + ": " + str(total[i]))

#squishy function
def squish(data) :
    return np.multiply(np.divide(data,255),2)-1

#batches
def miniBatch(data) :
    totBatches = []
    numBatches = int(len(data) / 256)
    leftover = len(data) % 256
    for i in range(numBatches) :
        totBatches.append(data[(i*256):((i+1)*256)])
    totBatches.append(data[(len(data)-leftover):])
    return totBatches

#hot labels
def hotLabels(labels) :
    encoded = np.zeros((len(labels[0]),10))
    for i in range(len(labels[0])) :
        encoded[i][labels[0][i]] = 1
    return encoded

#helpers for softmax
def subMax(arr) :
    return arr - max(arr)

def calcSoft(arr) :
    return np.divide(np.exp(arr),(np.sum(np.exp(arr))))

#softmax
def softMax(arr) :
    arr = np.apply_along_axis(subMax,1,arr)
    arr = np.apply_along_axis(calcSoft,1,arr)
    return arr

#cross entropy loss
def crossEntropyLoss(pred, actual) :
    return -np.sum(np.multiply(actual,np.log(pred)))/(actual.shape[0])

#accuracy
def accuracy(pred, actual) :
    size = len(pred)
    actual = actual.argmax(axis=1)
    pred = pred.argmax(axis=1)
    return np.divide(np.count_nonzero(actual == pred),size)

#backward pass
def backwardPass(data, label, weights1, biases1, weights2, biases2, y_hat, h, lRate) :
    #2nd set first (dldw2, dldb2)
    delta2 = np.divide(y_hat - label,len(y_hat))
    dldw2 = np.matmul(np.transpose(delta2),np.transpose(h))
    dldb2 = delta2.sum(axis=0)
    dldb2 = dldb2.reshape(10,1)

    #now onto the first set
    h[h > 0] = 1
    delta1 = np.multiply(np.matmul(np.transpose(weights2),np.transpose(delta2)),h)
    dldw1 = np.matmul(delta1,np.transpose(data))
    dldb1 = np.transpose(delta1).sum(axis=0)
    dldb1 = dldb1.reshape(30,1)

    #gradient descent
    return (weights1 - np.multiply(dldw1,lRate)), (biases1 - np.multiply(dldb1,lRate)), (weights2 - np.multiply(dldw2,lRate)), (biases2 - np.multiply(dldb2,lRate))

#onePass
def onePass(data, label, weights1, biases1, weights2, biases2, lrate) :
    #forwardPass
    #First factor in weights1 and biases1
    z = np.matmul(weights1,data) + biases1

    #next, reLu
    z[z<0]=0

    #Factor in weights2 and biases2
    a = np.matmul(weights2,z) + biases2
    a = np.transpose(a)
    #softmax
    a = softMax(a)
    #backwardPass
    return backwardPass(data, label, weights1, biases1, weights2, biases2, a, z, lrate)

#forward pass for loss and accuracy
def forwardPass(data, label, weights1, biases1, weights2, biases2) :
    #First factor in weights1 and biases1
    z = np.matmul(weights1,data) + biases1

    #next, reLu
    z[z<0]=0

    #Factor in weights2 and biases2
    a = np.matmul(weights2,z) + biases2
    a = np.transpose(a)

    #then softMax
    a = softMax(a)

    #return
    return crossEntropyLoss(a,label), accuracy(a,label)

#plot points
def plotPoints(lRate, t, v, pType, set) :
    epoch = []

    for i in range(len(t)) :
        epoch.append(i)

    plt.plot(epoch, t, label="training")
    plt.plot(epoch, v, label="validation")
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel(pType)
    plt.savefig(pType + "_" + str(set) + "_" + str(lRate) + ".png",)
    plt.clf()

def main() :
    print("program takes approximately 13 minutes to complete")
    xTrain, yTrain, xVal, yVal, xTest, yTest = readData()
    dataBalanced(yTrain)
    xTrain = squish(xTrain)
    xVal = squish(xVal)
    xTest = squish(xTest)
    yTrain = hotLabels(yTrain)
    yVal = hotLabels(yVal)
    yTest = hotLabels(yTest)
    dataBatches = miniBatch(xTrain)
    labelBatches = miniBatch(yTrain)
    lRates = [0.001, 0.01, 0.1, 1, 10]
    bestlRate = 0
    lowestLoss = 1000

    #determine learning rate
    for i in range(len(lRates)) :
        weights1 = 0.0001*np.random.randn(30,len(xTrain[0]))
        weights2 = 0.0001*np.random.randn(10,30)
        biases1 = np.zeros((30,1))
        biases2 = np.zeros((10,1))
        #for each epoch :
        tepochLoss = []
        tepochAccuracy = []
        vepochLoss = []
        vepochAccuracy = []
        #initial run
        loss, acc = forwardPass(np.transpose(xTrain), yTrain, weights1, biases1, weights2, biases2)
        tepochLoss.append(loss)
        tepochAccuracy.append(acc)
        loss, acc = forwardPass(np.transpose(xVal), yVal, weights1, biases1, weights2, biases2)
        vepochLoss.append(loss)
        vepochAccuracy.append(acc)
        for j in range(100) :
            #for each miniBatch :
            for k in range(len(dataBatches)) :
                weights1, biases1, weights2, biases2 = onePass(np.transpose(dataBatches[k]),labelBatches[k],weights1,biases1,weights2,biases2,lRates[i]) 
            #forward pass for training loss and accuracy
            loss, acc = forwardPass(np.transpose(xTrain), yTrain, weights1, biases1, weights2, biases2)
            tepochLoss.append(loss)
            tepochAccuracy.append(acc)
            #forward pass for validation
            loss, acc = forwardPass(np.transpose(xVal), yVal, weights1, biases1, weights2, biases2)
            vepochLoss.append(loss)
            vepochAccuracy.append(acc)
        
        #plot points
        plotPoints(lRates[i],tepochLoss,vepochLoss,"loss",1)
        plotPoints(lRates[i],tepochAccuracy,vepochAccuracy,"acc",1)

        #determine lowest validation loss
        if(lowestLoss > vepochLoss[-1]) :
            bestlRate = lRates[i]
            lowestLoss = vepochLoss[-1]
    
    #train data with optimal learning rate, implement early stopping
    weights1 = 0.0001*np.random.randn(30,len(xTrain[0]))
    weights2 = 0.0001*np.random.randn(10,30)
    biases1 = np.zeros((30,1))
    biases2 = np.zeros((10,1))

    tepochLoss = []
    tepochAccuracy = []
    vepochLoss = []
    vepochAccuracy = []

    for j in range(100) :
        #for each miniBatch :
        for k in range(len(dataBatches)) :
            weights1, biases1, weights2, biases2 = onePass(np.transpose(dataBatches[k]),labelBatches[k],weights1,biases1,weights2,biases2,bestlRate) 
        #forward pass for training loss and accuracy
        loss, acc = forwardPass(np.transpose(xTrain), yTrain, weights1, biases1, weights2, biases2)

        tepochLoss.append(loss)
        tepochAccuracy.append(acc)
        #forward pass for validation
        loss, acc = forwardPass(np.transpose(xVal), yVal, weights1, biases1, weights2, biases2)
        vepochLoss.append(loss)
        vepochAccuracy.append(acc)
        #early stopping condition
        if(vepochLoss[j] > vepochLoss[j-1] and tepochLoss[j] < tepochLoss[j-1]) :
            break
        
    plotPoints(lRates[i],tepochLoss,vepochLoss,"loss",2)
    plotPoints(lRates[i],tepochAccuracy,vepochAccuracy,"acc",2)
    
    print("training loss after training")
    print(tepochLoss[-1])
    print("validation loss after training")
    print(vepochLoss[-1])
    print("training accuracy after training")
    print(tepochAccuracy[-1])
    print("validation accuracy after training")
    print(vepochAccuracy[-1])
    
    #now run the testing data
    loss, acc = forwardPass(np.transpose(xTest), yTest, weights1, biases1, weights2, biases2)
    print("Sample cross entropy loss: " + str(loss))
    print("Sample accuracy: " + str(acc))
    
main()