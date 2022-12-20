from sklearn import preprocessing
import pandas as pd
import numpy as np


def NormalizeValues(csvFile):
    inputData = pd.read_csv(csvFile)
    inputDf = pd.DataFrame(inputData)
    normDf = pd.DataFrame(columns=['sepall', 'sepalw', 'petall', 'petalw', 'class'])
    # Normalize the data row by row and store them in a Dataframe.
    for row in inputDf.index.values:
        row_array = np.array(inputDf[['sepall', 'sepalw', 'petall', 'petalw']].iloc[row].values)
        normalized_data = preprocessing.normalize([row_array])
        normalizedArray = np.append(normalized_data,
                                    inputDf[['class']].iloc[row].values)  # Append 'class' column that doesn't change
        normDf.loc[row] = [normalizedArray[0], normalizedArray[1], normalizedArray[2], normalizedArray[3],
                           normalizedArray[4]]
    normDf.to_csv("normalized.csv", index=False)  # Write the result in a csv file.


def dist(a, b):
    d = [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    return sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3])


def ENN(normCsvFile, k):
    inputData = pd.read_csv(normCsvFile)
    inputDf = pd.DataFrame(inputData)

    # Calculate the distances
    distances = {}
    for idx in inputDf.index:
        distances[idx] = {}
        for idx2 in inputDf.index:
            distances[idx][idx2] = dist(inputDf.values[idx], inputDf.values[idx2])

    # Find the K-nearest neighbors
    for item in distances:
        tempDict = distances[item]
        sortedDict = sorted(tempDict.items(), key=lambda x: x[1])
        kNearestNeighbors = sortedDict[1:k + 1]
        # Find the major class ....


NormalizeValues('iris.csv')
