import numpy as np
import pandas as pd


# calculate the Entropy:
# Entropy(feature) = -sum(pi*log(pi))
def calcEnt(dataset):
    count0 = len(dataset[dataset['Species'] == 'Iris-setosa'])
    count1 = len(dataset[dataset['Species'] == 'Iris-versicolor'])
    count2 = len(dataset[dataset['Species'] == 'Iris-virginica'])
    all = len(dataset)
    p0 = count0 / all
    p1 = count1 / all
    p2 = count2 / all
    p = [p0,p1,p2]
    sum = 0
    for i in range(3):
        if p[i] == 0:
            continue
        sum += p[i]*np.log2(p[i])
    return -sum


# Gain(D,a) = Ent(D) - sum(|Di|/|D| * Ent(Di))
def calcGain(dataset,feature,Ta):
    oldEnt = calcEnt(dataset)
    oldCount = len(dataset)
    data0 = dataset[dataset[feature]<Ta]
    data1 = dataset[dataset[feature]>Ta]
    newCount0 = len(data0)
    newCount1 = len(data1)
    sum = 0
    sum += newCount0 / oldCount * calcEnt(data0)
    sum += newCount1 / oldCount * calcEnt(data1)
    return oldEnt - sum


# calculate the Ta , because of the continuous value
#https://blog.csdn.net/qq_40875866/article/details/79508854
def calcTa(values):
    newValues = []
    for i in range(len(values)-1):
        Ta = (values[i]+values[i+1]) / 2
        newValues.append(Ta)
    return newValues


# calculate the best Gain and the Ta.
def calcBestGain(dataset,feature):
    values = list(set(dataset[feature]))
    Tas = calcTa(values)
    bestGain = 0
    bestTa = 0
    for Ta in Tas:
        gain = calcGain(dataset,feature,Ta)
        if gain > bestGain:
            bestGain = gain
            bestTa = Ta
    return bestGain, bestTa


# choose the best feature of current dataset
def chooseBestFeature(dataset,features):
    bestGain = 0
    bestTa = 0
    bestFeature = features[0]
    for feature in features:
        gain,Ta = (calcBestGain(dataset,feature))
        if gain > bestGain:
            bestGain = gain
            bestTa = Ta
            bestFeature = feature
    return bestFeature,bestTa


# build decision tree
def buildTree(dataset,bestFeature,bestTa):
    root = Node(bestFeature)
    leftChildDataSet = dataset[dataset[bestFeature]<bestTa]
    rightChildDataSet = dataset[dataset[bestFeature]>bestTa]
    newFeatures = features.drop(bestFeature)
    bestFeature0,bestTa0 = chooseBestFeature(leftChildDataSet,newFeatures)
    bestFeature1,bestTa1 = chooseBestFeature(rightChildDataSet,newFeatures)
    root.left = buildTree(leftChildDataSet,bestFeature0,bestTa0)
    root.right = buildTree(rightChildDataSet,bestFeature1,bestTa1)
    return root


class Node:
    def __init__(self, val):
        self.right = None
        self.left = None
        self.data = val

if __name__ == '__main__':
    data = pd.read_csv("iris.data")
    loandata = pd.DataFrame(data)
    data = data.sample(
        frac=1.0)  # Return a random sample of items from an axis of object. frac: Fraction of axis items to return.
    data = data.reset_index(drop=True)  # Reset the index
    # print(data)
    trainDataSet = data.loc[0:119]
    testDataSet = data.loc[120:]
    features = data.columns[0:-1]
    bestFeature,bestTa = chooseBestFeature(trainDataSet,features)
    root = buildTree(trainDataSet,bestFeature,bestTa)
