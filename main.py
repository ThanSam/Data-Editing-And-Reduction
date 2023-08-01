from DataProcessing import NormalizeValues
from ENN import ENN
from IB2 import IB2


# Main Program
selectedDataset = "C:/Users/Sam/Documents/Data-Editing-and-Reduction/datasets/iris.csv"
NormalizeValues(selectedDataset)
ENN('normalized.csv', 3)  # Define: N = 3 (default For ENN)
IB2('normalized.csv')
