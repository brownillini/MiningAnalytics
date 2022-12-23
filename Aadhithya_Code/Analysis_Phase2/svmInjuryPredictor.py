import pandas as pd
import numpy as np

import re # for regex

import nltk # NLP toolkit
# nltk.download() # uncomment this when you run nltk for the first time
from sklearn.svm import SVC

import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.model_selection import GridSearchCV
 


##### CHANGE THESE VARIABLES TO GET THE RESULTS FOR TWO SEPARATE GROUND TRUTHS. PLEASE UNCOMMENT THE GROUND TRUTH YOU WANT TO CHECK OUT.

# flag = "test" # change the value to train if you wish to train svm
flag = "train"

#### END OF VARIABLES


### Function Definitions ###
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   
    ##characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()    
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    # back to string from list
    text = " ".join(lst_text)
    return text



### END OF FUNCTION DEFINITIONS ###

### Script begins ###
y_test =[]
y_pred = []
lst_stopwords = nltk.corpus.stopwords.words("english")


data = pd.read_csv('translated_removed_duplicates.csv', encoding = 'unicode_escape')
data = data.rename(columns={'Incident Description':'text'})
data['text_clean'] = data['text'].apply(lambda x: 
        utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
        lst_stopwords=lst_stopwords))
X_test = data['text_clean']
data['category'] = data['Risk Rating'].apply(lambda x: x.split('.')[0])

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['Risk Rating'], test_size=0.1, random_state=1773)


## END OF PREPROCESSING

categories = ['low', 'moderate', 'high', 'critical']

if flag == "train":
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    # pickle.dump(vectorizer, open("vectorizer.pickle", "wb")) #Save vectorizer
    pickle.dump(vectorizer, open("vectorizerTFIDFSVM.pickle", "wb")) #Save vectorizer
vectorizer = pickle.load(open("vectorizerTFIDFSVM.pickle", 'rb'))      #Load vectorizer
X_test = vectorizer.transform(X_test)

# for svm
print("SVM results")
filename = 'svmModelWithBestFit.sav'
if flag == "train":
    svm = SVC(probability=True, C = 1, gamma=1, kernel='linear')
    # The commented code was used for hyper parameter tuning
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf', 'linear']}
    # param_grid = {'C': [1],
    #         'gamma': [1],
    #         'kernel': ['linear']}
    # grid = GridSearchCV(SVC(probability=True), param_grid, refit = True, verbose = 3, cv=10) 
    # grid.fit(X_train, y_train)     # fitting the model for grid search
    # print(grid.best_params_)
    svm.fit(X_train, y_train)
    pickle.dump(svm, open(filename, 'wb')) # Save the model

loaded_model_svm = pickle.load(open(filename, 'rb'))
print("I am here")
y_pred = loaded_model_svm.predict(X_test)
labels = ['low', 'moderate', 'high', 'critical']   

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


