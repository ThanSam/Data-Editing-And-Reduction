from sklearn import preprocessing
import pandas as pd
import numpy as np
import math


# Calculate the Euclidean distance between 2 multidimensional points
def dist(a, b):

    distance = 0
    for i in range(0, len(a)):
        distance_i = abs(a[i] - b[i])
        distance += math.sqrt(distance_i * distance_i)
    return distance


# Calculate the distances between each dataset observation
def calculateDistances(inputDataFrame):

    columns = list(inputDataFrame.columns)
    columns.pop()  # Remove the 'class' column from calculating distances
    distances = []
    # for col in inputDataFrame.index:
    for i in inputDataFrame.index:
        i_distances = []
        for j in inputDataFrame.index:
            i_distances.append(dist(inputDataFrame[columns[:]].values[i],
                                    inputDataFrame[columns[:]].values[j]))
        distances.append(i_distances)
    return distances


# Returns the K-Nearest Neighbors indexes in a list
def findKNearestNeighbors(distances, K):

    distancesDict = {}
    for i in range(0, len(distances)):
        distancesDict[i] = distances[i]
    distancesDict = sorted(distancesDict.items(), key=lambda item: item[1])
    NearestNeighbors = []
    for i in range(1, K+1):
        NearestNeighbors.append(distancesDict[i][0])
    return NearestNeighbors


def findMajorityClass(ObservationsDf, NearestNeighbors):

    IrisSetosaCnt = IrisVersicolorCnt = IrisVirginicaCnt = 0

    for i in NearestNeighbors:
        if ObservationsDf.loc[i, "class"] == 'Iris-setosa':
            IrisSetosaCnt += 1
        elif ObservationsDf.loc[i, "class"] == 'Iris-versicolor':
            IrisVersicolorCnt += 1
        else:
            IrisVirginicaCnt += 1

    MajorityClassCnt = max(IrisSetosaCnt, IrisVersicolorCnt, IrisVirginicaCnt)
    if MajorityClassCnt == IrisSetosaCnt:
        return 'Iris-setosa'
    elif MajorityClassCnt == IrisVersicolorCnt:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'


def NormalizeValues(csvFile):

    inputData = pd.read_csv(csvFile)
    inputDf = pd.DataFrame(inputData)
    normDf = pd.DataFrame(columns=['sepall', 'sepalw', 'petall', 'petalw', 'class'])
    # Normalize the data row by row and store them in a Dataframe
    for row in inputDf.index.values:
        row_array = np.array(inputDf[['sepall', 'sepalw', 'petall', 'petalw']].iloc[row].values)
        normalized_data = preprocessing.normalize([row_array])
        normalizedArray = np.append(normalized_data,
                                    inputDf[['class']].iloc[row].values)  # Append 'class' column that doesn't change
        normDf.loc[row] = [normalizedArray[0], normalizedArray[1], normalizedArray[2], normalizedArray[3],
                           normalizedArray[4]]
    return normDf.to_csv("normalized.csv", index=False)  # Write the result in a csv file


def ENN(normCsvFile, K):

    inputData = pd.read_csv(normCsvFile)
    inputDf = pd.DataFrame(inputData)
    distances = calculateDistances(inputDf)
    deletedIndexes = []
    for i in range(len(distances)):  # For each observation of the dataset...
        NearestNeighbors = findKNearestNeighbors(distances[i], K)
        if findMajorityClass(inputDf, NearestNeighbors) != inputData.loc[i, 'class']:
            deletedIndexes.append(i)
    inputDf.drop(deletedIndexes, inplace=True)   # Delete the different classes observations
    inputDf.to_csv("reduced_ENN.csv", index=False)   # Write the result in a csv file


def IB2(normCsvFile):

    inputData = pd.read_csv(normCsvFile)
    trainingSet = pd.DataFrame(inputData)
    condensingSet = pd.DataFrame(columns=['sepall', 'sepalw', 'petall', 'petalw', 'class'])
    condensingSetIdx = 0

    #  Pick an item of TS and move it to CS
    tempDf = pd.DataFrame([trainingSet.iloc[0]])
    condensingSet = pd.concat([condensingSet, tempDf], axis=0, ignore_index=True)
    trainingSet.drop([0], inplace=True)

    for i in range(1, len(trainingSet)):
        tempDf = pd.DataFrame([trainingSet.iloc[i]])
        tempCondensingSet = pd.concat([condensingSet, tempDf], axis=0, ignore_index=True)
        distances = calculateDistances(tempCondensingSet)
        nearestNeighbor = findKNearestNeighbors(distances[condensingSetIdx], 1)    # Find the nearest neighbor of item (i) in CS (K=1)
        if trainingSet.loc[i, 'class'] != tempCondensingSet.loc[nearestNeighbor[0], 'class']:
            condensingSet = tempCondensingSet     # CS <- CS U (i)
            condensingSetIdx += 1
        trainingSet.drop([i], inplace=False)     # TS <- TS - (i)

    condensingSet.to_csv("reduced_IB2.csv", index=False)  # Write the result in a csv file


# Main Program

NormalizeValues('iris.csv')
ENN('normalized.csv', 3)  # Define: N = 3 (default For ENN)
IB2('normalized.csv')
