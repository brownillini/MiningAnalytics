## for data
from fileinput import filename
from json import load
import torch
import pandas as pd
import numpy as np
from sklearn import metrics, manifold
## for processing
import re
import nltk
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for w2v
import gensim
import gensim.downloader as gensim_api

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import nn

import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
 
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords


data1 = pd.read_csv('interaction.labeled.csv', encoding = 'unicode_escape')
data1 = data1.rename(columns={'DESC':'text'})
print(data1.head())


data = pd.read_csv('MSHA.injuries.small.csv', encoding = 'unicode_escape')
print(data.head())

for index, row in data.iterrows():
    if row["INJ_BODY_PART"] == "FINGER(S)/THUMB":
       row["INJ_BODY_PART"] = "HAND"
    if row["INJ_BODY_PART"] =="""EYE(S) OPTIC NERVE/VISON""":
        row["INJ_BODY_PART"] ="""EYE"""
    if row["INJ_BODY_PART"] =="""HAND (NOT WRIST OR FINGERS)""" :
        row["INJ_BODY_PART"] = """HAND"""
    if row["INJ_BODY_PART"] =="""FINGER(S)/THUMB""" :
        row["INJ_BODY_PART"] = """HAND"""
    if row["INJ_BODY_PART"] =="""WRIST""" :
        row["INJ_BODY_PART"] = """HAND"""
    if row["INJ_BODY_PART"] =="""ANKLE""" :
        row["INJ_BODY_PART"] =  """ANKLE"""
    if row["INJ_BODY_PART"] =="""KNEE/PATELLA""" :
        row["INJ_BODY_PART"] = """KNEE"""
    if row["INJ_BODY_PART"] =="""SHOULDERS (COLLARBONE/CLAVICLE/SCAPULA)""" :
        row["INJ_BODY_PART"] = """SHOULDER"""
    if row["INJ_BODY_PART"] =="""BACK (MUSCLES/SPINE/S-CORD/TAILBONE)""" :
        row["INJ_BODY_PART"] = """BACK"""
    if row["INJ_BODY_PART"] =="""FOREARM/ULNAR/RADIUS""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """ABDOMEN/INTERNAL ORGANS""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """HIPS (PELVIS/ORGANS/KIDNEYS/BUTTOCKS)""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """ELBOW""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """FOOT(NOT ANKLE/TOE)/TARSUS/METATARSUS""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """MOUTH/LIP/TEETH/TONGUE/THROAT/TASTE""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """SCALP""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """CHEST (RIBS/BREAST BONE/CHEST ORGNS)""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """LOWER LEG/TIBIA/FIBULA""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """NECK""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """JAW INCLUDE CHIN""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """TOE(S)/PHALANGES""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """EAR(S) INTERNAL & HEARING""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """UPPER ARM/HUMERUS""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """BRAIN""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """THIGH/FEMUR"""  :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """NOSE/NASAL PASSAGES/SINUS/SMELL"""  :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """EAR(S) EXTERNAL""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """SKULL""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """EAR(S) INTERNAL & EXTERNAL""" :
        row["INJ_BODY_PART"] = """OTHER"""

    

    if row["INJ_BODY_PART"] == """FACE,NEC""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """ARM,NEC""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] ==  """HEAD,NEC""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """LEG, NEC""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """TRUNK,NEC""":
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """BODY PARTS, NEC"""  :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """LOWER EXTREMITIES,NEC""" :
        row["INJ_BODY_PART"] = """OTHER"""
    if row["INJ_BODY_PART"] == """UPPER EXTREMITIES, NEC""":
        row["INJ_BODY_PART"] = """OTHER"""
    
    if row["INJ_BODY_PART"] == """BODY SYSTEMS""":
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """MULTIPLE PARTS (MORE THAN ONE MAJOR)""":
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """TRUNK, MULTIPLE PARTS""" :
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """UPPER EXTREMITIES, MULTIPLE""":
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """LOWER EXTREMITIES, MULTIPLE PARTS""":
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """FACE, MULTIPLE PARTS""" :
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """ARM, MULTIPLE PARTS""" :
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """HEAD, MULTIPLE PARTS""":
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """LEG, MULTIPLE PARTS""" :
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """NO VALUE FOUND""" :
        row["INJ_BODY_PART"] = """EXCLUDE"""
    if row["INJ_BODY_PART"] == """UNCLASSIFIED""" :
        row["INJ_BODY_PART"] = """EXCLUDE"""

for index, row in data.iterrows():
    if row["INJ_BODY_PART"] == """EXCLUDE""":
        data.drop(index, inplace=True)
data = data.rename(columns={'NARRATIVE':'text'})
print(data.sample(5))

mshaData = {"content": data['text'], "labels": data["INJ_BODY_PART"]}
mshaDf = pd.DataFrame(mshaData, columns=["content", "labels"])

heclaData = {"content": data1['text'], "labels": data1["Label (R Reed)"]}
heclaDf = pd.DataFrame(heclaData, columns=["content", "labels"])

trainDf = mshaDf
testDf = heclaDf
print("OG here")
print(testDf.head())
# trainDf = mshaDf.sample(frac=0.8, random_state=25)
# testDf = mshaDf.drop(trainDf.index)

wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

def tokenize_lemma_stopwords(text):
    text = str(text).replace("\n", " ")
    # split string into words (tokens)
    tokens = nltk.tokenize.word_tokenize(text.lower())
    # keep strings with only alphabets
    tokens = [t for t in tokens if t.isalpha()]
    # put words into base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
    tokens = [stemmer.stem(t) for t in tokens]
    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    cleanedText = " ".join(tokens)
    return cleanedText

def dataCleaning(df):
    data = df.copy()
    print(data["content"])
    data["content"] =data["content"].apply(tokenize_lemma_stopwords)
    return data
cleanedTrainData = dataCleaning(trainDf)
# cleanedTestData = dataCleaning(testDf)
cleanedTestData = dataCleaning(testDf)

vectorizer = TfidfVectorizer()
vectorised_train_documents = vectorizer.fit_transform(cleanedTrainData["content"])

pickle.dump(vectorizer, open("vectorizerSVMMulti.pickle", "wb")) #Save vectorizer
vectorizer =pickle.load(open("vectorizerSVMMulti.pickle", 'rb'))      #Load vectorizer
# pickle.dump(vectorizer, open("vectorizerRFMulti.pickle", "wb")) #Save vectorizer
# vectorizer =pickle.load(open("vectorizerRFMulti.pickle", 'rb'))      #Load vectorizer
vectorised_test_documents = vectorizer.transform(cleanedTestData["content"])

train_categories = []
test_categories = []
print(mshaDf)
for index, row in trainDf.iterrows():
    print(row["labels"])
    train_categories.append({row['labels']})
# for index, row in testDf.iterrows():
#     test_categories.append({row['labels']})
print("all labels")
# print(train_categories)
for index, row in testDf.iterrows():
    print(row['labels'])
    x = row['labels'].split('.')
    print(x)
    labelSet = set(x)
    print(labelSet)
    test_categories.append(labelSet)
# print("Test categories here")
# print(test_categories)
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_categories)
test_labels = mlb.transform(test_categories)
# for svm
print("SVM results")


svmClassifier = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svmClassifier.fit(vectorised_train_documents, train_labels)

svmPreds = svmClassifier.predict(vectorised_test_documents)

# # print(svmPreds)
 
# # grid = GridSearchCV(SVC(probability=True), param_grid, refit = True, verbose = 3)
 
# # fitting the model for grid search
# # grid.fit(X_train, y_train)


# # _ = svm.fit(X_train, y_train)

filename = 'svmModelMultiLabel.sav'
pickle.dump(svmClassifier, open(filename, 'wb'))
loaded_model_svm = pickle.load(open(filename, 'rb'))
svmPreds = loaded_model_svm.predict(vectorised_test_documents)
labels = ['ANKLE','BACK','EYE','HAND','KNEE','OTHER','SHOULDER']         

# rfClassifier = RandomForestClassifier(n_jobs=-1, n_estimators=10, random_state=0)
# rfClassifier.fit(vectorised_train_documents, train_labels)

# filename = 'rfMultiLabel.sav'
# pickle.dump(rfClassifier, open(filename, 'wb'))
# loaded_model_rf = pickle.load(open(filename, 'rb'))
# rfPreds = rfClassifier.predict(vectorised_test_documents)
# # loaded_model_svm = pickle.load(open(filename, 'rb'))
# # y_pred = loaded_model_svm.predict(X_test)
# # print(y_pred)
# # print(X_test.get_shape())
# # probs = loaded_model_svm.predict_proba(X_test)
# # k = 2
# # best_n = np.argsort(-probs, axis=1)[:, :k]
# # for i in range(0, best_n.shape[0]):
# #     for j in range(0,k):
# #         print(labels[best_n.item((i,j))], end='.')
# #     print("\n")
# # # print(best_n.shape)
# # for i in range(0, X_test.get_shape()[0]):
# #     results = loaded_model_svm.predict_proba(X_test)[i]
# #     prob_per_class_dictionary = dict(zip(loaded_model_svm.classes_, results))
# #     # print(results)
# #     with open('outputnewSVM.txt', 'a+') as f:
# #         f.write(str(results))
#     # print(prob_per_class_dictionary)

# # print(classification_report(y_test, y_pred))
# # print(confusion_matrix(y_test, y_pred))
print(classification_report(test_labels, svmPreds))
# print(test_labels)
print(test_labels)
print(svmPreds)
np.savetxt('aaatest.out', svmPreds)
np.savetxt('aaatest.out', test_labels)
# grid_predictions = grid.predict(X_test)
 
# print classification report
# print(classification_report(y_test, grid_predictions))






