import numpy as np
from Directories import data
# Load data set and code labels as 0 = ’NO’, 1 = ’DH’, 2 = ’SL’
labels = [b'NO', b'DH', b'SL']
data = np.loadtxt(data, converters={6: lambda s: labels.index(s)})

# Separate features from labels
x = data[:,0:6]
y = data[:,6]

# Divide into training and test set
training_indices = list(range(0,20)) + list(range(40,188)) + list(range(230,310))
test_indices = list(range(20,40)) + list(range(188,230))

trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]

###Creating the individual functions to calculate L2 Norm
def squared_dist(x,y):
    #sum of squared differences
    return np.sum(np.square(x-y))

def find_NN(x):
    #compute the distances from x to every row in the train data
    distances = [squared_dist(x, trainx[i,]) for i in range(len(trainy))]
    #return the index # of the minimum value in the 'distances' array
    return np.argmin(distances)

def NN_classifier(x):
    #Get the index of the the nearest neighbor
    index = find_NN(x)
    #Get the class of the index
    return trainy[index]

#testing the result of our invidiual functions/components as a baseline
test_predictions = [NN_classifier(testx[i,]) for i in range(len(testy))]
print(test_predictions)


###Putting them all together, with L1 & L2 Norm, and outputting an array of classifications
def NN_LN(trainx, trainy, testx):
    L1IndexArray = []
    L2IndexArray = []
    for x in range(len(testx)):
        L1Distance = [np.sum(abs((testx[x,]-trainx[i,]))) for i in range(len(trainy))]
        L1Index = np.argmin(L1Distance)
        L1Label = trainy[L1Index]
        L1IndexArray.append(L1Label)
        
        L2Distance = [np.sum((testx[x,]-trainx[i,])**2) for i in range(len(trainy))]      
        L2Index = np.argmin(L2Distance)        
        L2Label = trainy[L2Index]        
        L2IndexArray.append(L2Label)
    return L1IndexArray, L2IndexArray

testy_L1, testy_L2 = NN_LN(trainx, trainy, testx)

#L2 matches our baseline
print(testy_L1)
print(testy_L2)

###constructing a confusion matrix in Numpy
def confusion(testy, testy_fit):
    Categories = np.unique(testy)
    CMatrix = np.zeros((len(Categories), len(Categories)))
    for x in range(len(Categories)):
        for i in range(len(Categories)):
                CMatrix[x, i] = np.sum((testy == Categories[x]) & (testy_fit == Categories[i]))
    return CMatrix

ConfusionMatrixL1 = confusion(testy, testy_L1)
ConfusionMatrixL2 = confusion(testy, testy_L2)

print(ConfusionMatrixL1)
print(ConfusionMatrixL2)

###Interpreting the confusion matrix

#testy data consists of true labels and testy_L1 and testy_L2 are predicted labels. In testy, there are 20 NP classifications and, for 
#example, in testy_L1 #there are 26 NP classifications. We can see that NP is the most misclassified label in our L1 prediction set. 
#We can also see that both the models correctly predicted the SP classifier. Lastly, we can see that L1 and L2 differ in classification
#of a single test point. L1 (incorrectly) classifies an extra DH while L2 (correctly) classifies an extra NP. 