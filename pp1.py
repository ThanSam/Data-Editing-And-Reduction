from sklearn import preprocessing
import pandas as pd
import numpy as np
import math


# Calculate the Euclidean distance between 2 multidimensional points
def dist(a, b):
    distance = 0
    for i in range(0, len(a)):
        distance_i = a[i] - b[i]
        distance += math.sqrt(distance_i * distance_i)
    return distance


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
    return normDf.to_csv("normalized.csv", index=False)  # Write the result in a csv file.


def ENN(normCsvFile, k):
    inputData = pd.read_csv(normCsvFile)
    inputDf = pd.DataFrame(inputData)

    # Calculate the distances
    distances = []

    for i in inputDf.index:
        i_distances = []
        for j in inputDf.index:
            i_distances.append(dist([inputDf['sepall'][i], inputDf['sepalw'][i], inputDf['petall'][i], inputDf['petalw'][i]],
                                    [inputDf['sepall'][j], inputDf['sepalw'][j], inputDf['petall'][j], inputDf['petalw'][j]]))
        distances.append(i_distances)

    print(distances)


NormalizeValues('iris.csv')
ENN('normalized.csv', 3)  # Define: N = 3
