# Data Editing and Reduction
A project for the univeristy lesson of "Knowledge Discovery from Databases". 

Using as training datasets csv files, comprised of numerical attributes only with the exception of the class attribute, are implemented in Python 3.9 the following operations:
<br>
<br>
* __NormalizeValues(inputCsvFile):__ &nbsp;Normalizing the values of all attributes-except for the class attribute(i.e., transforms them in the [0,1] range).

<br>

* __ENN(inputNormalizedCsvFile, K):__  &nbsp;Takes as input a normalized csv file and the required algorithm parameter K. Applies the editing algorithm ENN on it.

<br>

* __IB2(inputNormalizedCsvFile):__  &nbsp;Takes as input a normalized csv file and applies the instance reduction algorithm IB2 on it.

<br>

The code is tested with the "iris.csv" and "letter-recognition.csv" datasets. <br>The output in each program is written to a csv file as well.


