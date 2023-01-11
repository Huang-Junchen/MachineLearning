'''
@Description:
@Author     :Junchen Huang
@Date       :2023/01/11 15:44:31
'''

from math import log
import operator

def calcShannonEnt(dataSet):
    """
    @description: 计算香农熵
    @param      : dataSet 数据集
    @Returns    : shannonEnt 香农熵
    """
    
    numEntries = len(dataSet)
    labelCounts = {}
    
    '''为所有可能的分类创建字典'''
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    '''通过公式计算香农熵'''
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    
    return shannonEnt
    
def createDataSet():
    """
    @description: 创建简单的数据集用于测试函数calcShannonEnt(dataSet)
    @param      : None
    @Returns    : dataSet, labels 数据集和标签
    """
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """
    @description: 按照给定特征划分数据集
    @param      : dataSet, axis, value 待划分的数据集、划分数据集的特征、需要返回的特征的值
    @Returns    : retDataSet 符合特征的子数据集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    @description: 选择最好的数据集划分方式
    @param      : dataSet 数据集
    @Returns    : bestFeature 最好的information gain
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        # 利用推导式取出该特征左右可能存在的值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)

        # 计算每种划分方式的信息熵
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain :
            bestInfoGain = infoGain
            bestFeature = i
    
    return bestFeature

def majorityCnt(classList):
    """
    @description: 多数表决来定义不唯一的类标签
    @param      : classList 叶节点数据集
    @Returns    : 
    """
    
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),\
        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    @description: 递归生成决策树
    @param      : dataSet labels 数据集和标签
    @Returns    : myTree 决策树
    """
    
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    # 遍历玩所有特征后返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}} # Python用字典来实现树
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        '''因为python中列表在函数中可被修改，因此每次需要创建新子标签列表'''
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
            (dataSet, bestFeat, value), subLabels)
    
    return myTree

    
    
if __name__ == '__main__':
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    print(myTree)

