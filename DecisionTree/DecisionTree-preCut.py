import numpy as np
import pandas as pd
import collections
from math import log
import operator


class Node():
    def __init__(self):
        self.feature = None
        self.value = None
        self.leftChild = None
        self.rightChild = None
        self.label = None


# calculate the Entropy:
# Entropy(feature) = -sum(pi*log(pi))
def calcShannonEnt(dataset):
    # 计算给定数据集的香农熵
    # param：dataset
    # return:
    # 计算数据总数
    all = len(dataset)

    # 统计标签
    labelCounts = collections.defaultdict(int)

    # 循环整个数据集，得到数据的分类标签
    for featureVec in dataset:
        # 得到当前的标签
        currentLabel = featureVec[-1]

        # 将对应的标签值加一
        labelCounts[currentLabel] += 1

    # 默认的信息熵
    shannonEnt = 0.0

    for key in labelCounts:
        # 计算出当前分类标签占总标签的比例数
        prob = float(labelCounts[key]) / all

        # 以2为底求对数
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def splitDataSetForSeries(dataSet, axis, value):
    """
    按照给定的数值，将数据集分为不大于和大于两部分
    :param dataSet: 要划分的数据集
    :param axis: 特征值所在的下标
    :param value: 划分值
    :return:
    """
    # 用来保存不大于划分值的集合
    eltDataSet = []
    # 用来保存大于划分值的集合
    gtDataSet = []
    # 进行划分，保留该特征值
    for feat in dataSet:
        if feat[axis] <= value:
            eltDataSet.append(feat)
        else:
            gtDataSet.append(feat)

    return eltDataSet, gtDataSet


def splitDataSet(dataSet, axis, value):
    """
    按照给定的特征值，将数据集划分
    :param dataSet: 数据集
    :param axis: 给定特征值的坐标
    :param value: 给定特征值满足的条件，只有给定特征值等于这个value的时候才会返回
    :return:
    """
    # 创建一个新的列表，防止对原来的列表进行修改
    retDataSet = []

    # 遍历整个数据集
    for featVec in dataSet:
        # 如果给定特征值等于想要的特征值
        if featVec[axis] == value:
            # 将该特征值前面的内容保存起来
            reducedFeatVec = featVec[:axis]
            # 将该特征值后面的内容保存起来，所以将给定特征值给去掉了
            reducedFeatVec.extend(featVec[axis + 1:])
            # 添加到返回列表中
            retDataSet.append(reducedFeatVec)

    return retDataSet


def calcInfoGainForSeries(dataSet, i, baseEntropy):
    """
    计算连续值的信息增益
    :param dataSet:整个数据集
    :param i: 对应的特征值下标
    :param baseEntropy: 基础信息熵
    :return: 返回一个信息增益值，和当前的划分点
    """

    # 记录最大的信息增益
    maxInfoGain = 0.0

    # 最好的划分点
    bestMid = -1

    # 得到数据集中所有的当前特征值列表
    featList = [example[i] for example in dataSet]

    # 得到分类列表
    classList = [example[-1] for example in dataSet]

    dictList = dict(zip(featList, classList))

    # 将其从小到大排序，按照连续值的大小排列
    sortedFeatList = sorted(dictList.items(), key=operator.itemgetter(0))

    # 计算连续值有多少个
    numberForFeatList = len(sortedFeatList)

    # 计算划分点，保留三位小数
    midFeatList = [round((sortedFeatList[i][0] + sortedFeatList[i+1][0])/2.0, 3)for i in range(numberForFeatList - 1)]

    # 计算出各个划分点信息增益
    for mid in midFeatList:
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, i, mid)

        # 计算两部分的特征值熵和权重的乘积之和
        gtEnt = calcShannonEnt(gtDataSet)

        newEntropy = len(eltDataSet)/len(featList)*calcShannonEnt(eltDataSet) + len(gtDataSet)/len(featList)*calcShannonEnt(gtDataSet)

        # 计算出信息增益
        infoGain = baseEntropy - newEntropy
        #print('当前划分值为：' + str(mid) + '，此时的信息增益为：' + str(infoGain))
        if infoGain > maxInfoGain:
            bestMid = mid
            maxInfoGain = infoGain

    return maxInfoGain, bestMid


def calcInfoGain(dataSet ,featList, i, baseEntropy):
    """
    计算信息增益
    :param dataSet: 数据集
    :param featList: 当前特征列表
    :param i: 当前特征值下标
    :param baseEntropy: 基础信息熵
    :return:
    """
    # 将当前特征唯一化，也就是说当前特征值中共有多少种
    uniqueVals = set(featList)

    # 新的熵，代表当前特征值的熵
    newEntropy = 0.0

    # 遍历现在有的特征的可能性
    for value in uniqueVals:
        # 在全部数据集的当前特征位置上，找到该特征值等于当前值的集合
        subDataSet = splitDataSet(dataSet=dataSet, axis=i, value=value)
        # 计算出权重
        prob = len(subDataSet) / float(len(dataSet))
        # 计算出当前特征值的熵
        newEntropy += prob * calcShannonEnt(subDataSet)

    # 计算出“信息增益”
    infoGain = baseEntropy - newEntropy

    return infoGain


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分特征，根据信息增益值来计算，可处理连续值
    :param dataSet:
    :return:
    """
    # 得到数据的特征值总数
    numFeatures = len(dataSet[0]) - 1

    # 计算出基础信息熵
    baseEntropy = calcShannonEnt(dataSet)

    # 基础信息增益为0.0
    bestInfoGain = 0.0

    # 最好的特征值
    bestFeature = -1

    # 标记当前最好的特征值是不是连续值
    flagSeries = 0

    # 如果是连续值的话，用来记录连续值的划分点
    bestSeriesMid = 0.0

    # 对每个特征值进行求信息熵
    for i in range(numFeatures):

        # 得到数据集中所有的当前特征值列表
        featList = [example[i] for example in dataSet]

        if isinstance(featList[0], str):
            infoGain = calcInfoGain(dataSet, featList, i, baseEntropy)
        else:
            # print('当前划分属性为：' + str(labels[i]))
            infoGain, bestMid = calcInfoGainForSeries(dataSet, i, baseEntropy)

        # print('当前特征值为：' + labels[i] + '，对应的信息增益值为：' + str(infoGain))

        # 如果当前的信息增益比原来的大
        if infoGain > bestInfoGain:
            # 最好的信息增益
            bestInfoGain = infoGain
            # 新的最好的用来划分的特征值
            bestFeature = i

            flagSeries = 0
            if not isinstance(dataSet[0][bestFeature], str):
                flagSeries = 1
                bestSeriesMid = bestMid

    # print('信息增益最大的特征为：' + labels[bestFeature])
    if flagSeries:
        return bestFeature, bestSeriesMid
    else:
        return bestFeature


def createDataSet():
    data = pd.read_csv("iris.data")
    loandata = pd.DataFrame(data)
    data = data.sample(
        frac=1.0)  # Return a random sample of items from an axis of object. frac: Fraction of axis items to return.
    data = data.reset_index(drop=True)  # Reset the index
    # print(data)
    trainDataSet = data.loc[:119]
    testDataSet = data.loc[120:]
    # 特征值列表
    labels = data.columns[0:-1]

    return np.array(trainDataSet), np.array(testDataSet), np.array(labels)


def majorityCnt(classList):
    """
    找到次数最多的类别标签
    :param classList:
    :return:
    """
    # 用来统计标签的票数
    classCount = collections.defaultdict(int)

    # 遍历所有的标签类别
    for vote in classList:
        classCount[vote] += 1

    # 从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回次数最多的标签
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 特征标签
    :return:
    """
    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]

    # 统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if classList.count(classList[0]) == len(classList):
        node = Node()
        node.label = classList[0]
        return node

    # 如果数据在所有属性上的取值全部相等，就无法继续划分，此时停止划分
    # 取第一行的数据的所有特征，与其他所有数据的特征比较，如果相等则说明取值全部相等，此时停止划分
    # 相等标志
    equalFlag = 1
    valueList = dataSet[0][:-1]
    for instance in dataSet:
        # 如果有不相等的，相等标志置为0
        if not (valueList==instance[:-1]).all():
            equalFlag = 0
            break
    # 如果全部相等，那么返回占大多数的类别
    if equalFlag:
        node = Node()
        node.label = majorityCnt(classList)
        return node

    # 计算第一行有多少个数据，如果只有一个的话说明所有的特征属性都遍历完了，剩下的一个就是类别标签
    if len(dataSet[0]) == 1:
        # 返回剩下标签中出现次数较多的那个
        node = Node()
        node.label = majorityCnt(classList)
        return node

    # 选择最好的划分特征，得到该特征的下标
    bestFeat = chooseBestFeatureToSplit(dataSet)

    # 得到最好特征的名称
    bestFeatLabel = ''

    # 记录此刻是连续值还是离散值,1连续，2离散
    flagSeries = 0

    # 如果是连续值，记录连续值的划分点
    midSeries = 0.0

    # 如果是元组的话，说明此时是连续值
    if isinstance(bestFeat, tuple):
        root = Node()
        root.feature = bestFeat[0]
        root.value = bestFeat[1]
        # 重新修改分叉点信息
        bestFeatLabel = str(labels[bestFeat[0]]) + '小于' + str(bestFeat[1]) + '?'
        # 得到当前的划分点
        midSeries = bestFeat[1]
        # 得到下标值
        bestFeat = bestFeat[0]
        # 连续值标志
        flagSeries = 1
    else:
        # 得到分叉点信息
        bestFeatLabel = labels[bestFeat]
        # 离散值标志
        flagSeries = 0

    # 使用一个字典来存储树结构，分叉处为划分的特征名称
    # myTree = {bestFeatLabel: {}}

    # 得到当前特征标签的所有可能值
    featValues = [example[bestFeat] for example in dataSet]

    # 连续值处理
    if flagSeries:
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, bestFeat, midSeries)
        # 得到剩下的特征标签
        subLabels = labels[:]
        # 递归处理小于划分点的子树
        # subTree = createTree(eltDataSet, subLabels)
        # myTree[bestFeatLabel]['小于'] = subTree
        root.leftChild = createTree(eltDataSet,subLabels)

        # 递归处理大于当前划分点的子树
        # subTree = createTree(gtDataSet, subLabels)
        # myTree[bestFeatLabel]['大于'] = subTree
        root.rightChild = createTree(gtDataSet,subLabels)

        return root


# 预剪枝
def preCut(root,dataSet,testDataSet,labels,preCorrectRadio):

    # 如果全部都是一个类别，那么剪枝结束
    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]

    # 统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if classList.count(classList[0]) == len(classList):
        root.label = classList[0]
        return

    # 如果数据在所有属性上的取值全部相等，就无法继续划分，此时停止划分
    # 取第一行的数据的所有特征，与其他所有数据的特征比较，如果相等则说明取值全部相等，此时停止划分
    # 相等标志
    equalFlag = 1
    valueList = dataSet[0][:-1]
    for instance in dataSet:
        # 如果有不相等的，相等标志置为0
        if not (valueList == instance[:-1]).all():
            equalFlag = 0
            break
    # 如果全部相等，那么返回占大多数的类别
    if equalFlag:
        root.label = majorityCnt(classList)
        return

    # 选择最好的划分特征，得到该特征的下标
    bestFeat = chooseBestFeatureToSplit(dataSet)

    # 如果是元组的话，说明此时是连续值
    if isinstance(bestFeat, tuple):
        root.feature = bestFeat[0]
        root.value = bestFeat[1]
        root.label = None

        # 暂时保存root的类别
        label = root.label

        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, root.feature, root.value)

        # 拿到所有数据集的分类标签
        eltClassList = [example[-1] for example in eltDataSet]
        leftNode = Node()
        leftNode.label = majorityCnt(eltClassList)
        root.leftChild = leftNode

        # 拿到所有数据集的分类标签
        gtClassList = [example[-1] for example in gtDataSet]
        rightNode = Node()
        rightNode.label = majorityCnt(gtClassList)
        root.rightChild = rightNode

        # 计算测试集的正确率
        correctCount = 0
        for instance in testDataSet:
            label = testTree(root,instance)
            if instance[-1] == label:
                correctCount += 1
        newCorrectRadio = correctCount / len(testDataSet)

        # 比较正确率的大小
        #如果正确率增加 ，那么就进行划分
        if newCorrectRadio > preCorrectRadio:
            preCut(root.leftChild,eltDataSet,testDataSet,labels,newCorrectRadio)
            preCut(root.rightChild,gtDataSet,testDataSet,labels,newCorrectRadio)
        # 否则，就不进行划分
        else:
            root.label = label
            root.leftChild = None
            root.rightChild = None
            root.feature = None
            root.value = None


# 打印树
def printTree(root):
    if root.label is None:
        print("feature", root.feature)
        print('value', root.value)
    else:
        print("label",root.label)
        return
    printTree(root.rightChild)
    printTree(root.leftChild)

# 测试树
def testTree(tree,data):
    if tree.label is not None:
        return tree.label
    if data[tree.feature] > tree.value:
        return testTree(tree.rightChild,data)
    else:
        return testTree(tree.leftChild,data)


# 计算测试集上的正确率：
def calcTestCorrectRadio(root, testDataSet):
    correctCount = 0
    for instance in testDataSet:
        label = testTree(root, instance)
        if instance[-1] == label:
            correctCount += 1
    correctCountRadio = correctCount / len(testDataSet)
    return correctCountRadio


if __name__ == '__main__':
    """
    处理连续值时候的决策树
    """
    dataSet, testDataSet, labels,  = createDataSet()
    # 测试集总数
    testDataSetTotal = len(testDataSet)
    # 创建树根
    root = Node()
    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]
    root.label = majorityCnt(classList)
    # 统计测试集正确分类数
    correctCountRadio = calcTestCorrectRadio(root, testDataSet)
    # 开始剪枝
    preCut(root,dataSet,testDataSet,labels,correctCountRadio)
    correctCountRadio = calcTestCorrectRadio(root,testDataSet)
    print(correctCountRadio)