from DataProcessing import calculateDistances, findKNearestNeighbors
import pandas as pd


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
        nearestNeighbor = findKNearestNeighbors(distances[condensingSetIdx],
                                                1)  # Find the nearest neighbor of item (i) in CS (K=1)
        if trainingSet.loc[i, 'class'] != tempCondensingSet.loc[nearestNeighbor[0], 'class']:
            condensingSet = tempCondensingSet  # CS <- CS U (i)
            condensingSetIdx += 1
        trainingSet.drop([i], inplace=False)  # TS <- TS - (i)

    condensingSet.to_csv("reduced_IB2.csv", index=False)  # Write the result in a csv file
