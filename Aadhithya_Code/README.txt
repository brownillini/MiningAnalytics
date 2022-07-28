Here are the steps to reproduce the results for our preliminary analysis. 

1. Create Virtual environment using conda or virtualenv. 
2. Use Python version 3.8.8 for BERT and 3.7.2 for the remaining three algorithms.
3. Run pip install -r requirements.txt to install all the libraries.
4. Download all the required input files from https://arizona.app.box.com/folder/166545016111. 
5. Download the models and the vectorizers from https://arizona.app.box.com/folder/166541817819?s=l02uzcw1hnbyue87uvmjlz3lzs9fxs1y. You can find the relevant models and vectorizers within each folder for each model. 
5. Run the code.
6. Two output files will be generated. One output file contains 4 predicted labels for each input. The other output file contains the Precision, Recall and F1-scores for K=1.
7. Copy the output labels and paste it in the relevant [model]k4.csv file under the preds column. You can find these files here https://arizona.app.box.com/folder/166542314955. 
8. Run the UltimatePrecisionAndRecallCalculation.py file for calculating the Precision, Recall and F1-scores for K=4.
9. Repeat steps from 5 to 9 with different values and parameters for trial and error. 

P.S svmAndRFMultiLabel.py is future use. 