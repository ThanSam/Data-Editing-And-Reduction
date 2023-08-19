from sklearn import preprocessing
import pandas as pd
import numpy as np
import math


def NormalizeValues(csvFile):
    inputData = pd.read_csv(csvFile)
    inputDf = pd.DataFrame(inputData)
    dfColumns = list(inputDf.columns)
    normDf = pd.DataFrame(columns=dfColumns)
    # Normalize the data row by row and store them in a Dataframe
    for row in inputDf.index.values:
        row_array = np.array(inputDf[dfColumns[0:len(dfColumns) - 1]].iloc[row].values)
        normalized_data = preprocessing.normalize([row_array])
        normalizedArray = np.append(normalized_data,
                                    inputDf[['class']].iloc[row].values)  # Append 'class' column that doesn't change
        normDf.loc[row] = list(normalizedArray[0:len(normalizedArray)])
    return normDf.to_csv("normalized.csv", index=False)  # Write the result in a csv file


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
    for i in range(1, K + 1):
        NearestNeighbors.append(distancesDict[i][0])
    return NearestNeighbors


def findMajorityClass(ObservationsDf, NearestNeighbors):
    classes = ObservationsDf['class'].unique()  # Keep the different classes
    counter = [0] * len(classes)

    for i in NearestNeighbors:
        for cnt in range(len(classes)):
            if ObservationsDf.loc[i, 'class'] == classes[cnt]:
                counter[cnt] += 1

    majorityClassIndex = counter.index(max(counter))
    return classes[majorityClassIndex]
