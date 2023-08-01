from DataProcessing import calculateDistances, findKNearestNeighbors, findMajorityClass
import pandas as pd


def ENN(normCsvFile, K):
    inputData = pd.read_csv(normCsvFile)
    inputDf = pd.DataFrame(inputData)
    distances = calculateDistances(inputDf)
    deletedIndexes = []
    for i in range(len(distances)):  # For each observation of the dataset...
        NearestNeighbors = findKNearestNeighbors(distances[i], K)
        if findMajorityClass(inputDf, NearestNeighbors) != inputData.loc[i, 'class']:
            deletedIndexes.append(i)
    inputDf.drop(deletedIndexes, inplace=True)  # Delete the different classes observations
    inputDf.to_csv("reduced_ENN.csv", index=False)  # Write the result in a csv file
