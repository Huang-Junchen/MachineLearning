'''
@Description:
@Author     :Junchen Huang
@Date       :2023/01/11 15:44:31
'''

from math import log
import operator
import matplotlib.pyplot as plt
import pickle

'''定义文本框和箭头样式，用于绘制树形图'''
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

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
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        '''因为python中列表在函数中可被修改，因此每次需要创建新子标签列表'''
        subLabels = labels[:]
        del(subLabels[bestFeat])
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
            (dataSet, bestFeat, value), subLabels)
    
    return myTree

'''绘制树形图'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,\
        xycoords='axes fraction', xytext=centerPt,\
        textcoords='axes fraction', va="center",\
        ha="center", bbox=nodeType, arrowprops=arrow_args)

def getNumLeafs(myTree):
    """
    @description: 获取叶节点个数
    @param      : myTree 决策树
    @Returns    : numLeafs 叶节点个数
    """
    
    numLeafs = 0
    '''keys()函数返回的是dict_keys 对象，不再是list对象，需要强制转换'''
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    @description: 获取树的层数
    @param      : myTree 决策树
    @Returns    : maxDepth 树的层数
    """
    
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree)[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def classify(inputTree, featLabels, testVec):
    """
    @description: 使用决策树的分类函数
    @param      : inputTree, featLabel, testVec
    @Returns    : classLabel
    """
    
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    
    return classLabel

def storeTree(inputTree, filename):
    """
    @description: 存储决策树
    @param      : inputTree filename
    @Returns    : None
    """

    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close

def grabTree(filename):
    """
    @description: 读取存储的决策树
    @param      : filename
    @Returns    : 决策树
    """
    
    fr = open(filename)
    return pickle.load(fr)

if __name__ == '__main__':
    '''使用决策树预测隐形眼镜类型'''
    #myDat, labels = createDataSet()
    #myTree = createTree(myDat, labels)
    #print(myTree)
    #createPlot(myTree)
    #print(classify(myTree, labels, [1, 0]))
    #print(classify(myTree, labels, [1, 1]))
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    createPlot(lensesTree)

