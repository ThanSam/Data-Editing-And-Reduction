

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
        normalizedArray = np.append(normalized_data, inputDf[['class']].iloc[row].values)  # Append 'class' column that doesn't change
        normDf.loc[row] = [normalizedArray[0], normalizedArray[1], normalizedArray[2], normalizedArray[3],
                           normalizedArray[4]]
    normDf.to_csv("normalized.csv", index=False)  # Write the result in a csv file.


NormalizeValues('iris.csv')