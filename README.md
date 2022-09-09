# Mining Analytics

## Here are the steps to reproduce the results for our preliminary analysis. 

1. Create Virtual environment using conda or virtualenv. 
2. Use Python version 3.8.8 for BERT and 3.7.2 for the remaining three algorithms.
3. Run pip install -r requirements.txt to install all the libraries.
4. Download all the required input files from https://arizona.app.box.com/folder/166545016111. 
5. Download the models and the vectorizers from https://arizona.app.box.com/folder/166541817819?s=l02uzcw1hnbyue87uvmjlz3lzs9fxs1y. You can find the relevant models and vectorizers within each folder for each model. 
5. The code is in test mode by default. Run the code to get results for all three experts.
6. If you wish to train the models again, change the "flag" to "train" and the model training should occur.
7. FastText will by default take 5 minutes to autotune for hyperparameter tuning. You can change the time by setting the autotuneDuration parameter. 
8. BERT needs to be run on a GPU. It will take atleast 4 hours to train the model on 50,000 rows.
9. For SVM, sklearn.SVC is used with a linear kernel. It was run overnight on an 8GB RAM with 10 CPU cores for training on 50,000 rows. For 5,000 rows the training time drastically reduces.
10. usefulness correlation.py can be used to get the scores for association rules

P.S svmAndRFMultiLabel.py is future use. 
