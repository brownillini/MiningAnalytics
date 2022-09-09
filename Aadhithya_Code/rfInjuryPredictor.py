import pandas as pd
import numpy as np

import re # for regex

import nltk # NLP toolkit
# nltk.download() # uncomment this when you run nltk for the first time

from sklearn.ensemble import RandomForestClassifier
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
import average_precision

##### CHANGE THESE VARIABLES TO GET THE RESULTS FOR TWO SEPARATE GROUND TRUTHS. PLEASE UNCOMMENT THE GROUND TRUTH YOU WANT TO CHECK OUT.

# GROUND_TRUTH = 'Label (L Brown)'
# GROUND_TRUTH = 'Label (R Reed)'

GROUND_TRUTH_LIST = ['Label (Glenna)', 'Label (R Reed)', 'Label (L Brown)']
# Please change ground_truth to MSHA when in training mode

# GROUND_TRUTH = 'MSHA' # this is set when training the model
flag = "test" # change the value to train if you wish to train rf
# flag = "train"

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


def oneHotEncoding(Y_TRUE, Y_PRED, model, groundTruthIndex, comparisonIndex):
        df = pd.read_csv(model, encoding = 'unicode_escape')
        for index, row in df.iterrows():
            y_true = []
            y_scores = []
            classes = ['ankle', 'back', 'eye', 'hand', 'knee', 'other', 'shoulder']
            for bodyPart in classes:
                if str(row[groundTruthIndex]).lower().find(bodyPart) != -1:
                    y_true.append(1)
                if str(row[comparisonIndex]).lower().find(bodyPart) != -1:
                    y_scores.append(1)
                if str(row[groundTruthIndex]).lower().find(bodyPart) == -1:
                    y_true.append(0)
                if str(row[comparisonIndex]).lower().find(bodyPart) == -1:
                    y_scores.append(0)
            Y_TRUE.append(y_true)
            Y_PRED.append(y_scores)


### END OF FUNCTION DEFINITIONS ###

### Script begins ###
y_test =[]
y_test_labels = {'Label (Glenna)':[], 'Label (R Reed)':[], 'Label (L Brown)':[]}
y_pred = []
lst_stopwords = nltk.corpus.stopwords.words("english")

if flag == "test":
    data = pd.read_csv('interaction.labeled.csv', encoding = 'unicode_escape')
    data = data.rename(columns={'DESC':'text'})
    data['text_clean'] = data['text'].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))
    X_test = data['text_clean']
else:
    np.random.seed(112)
    data = pd.read_csv("""MSHA.injuries.csv""", encoding= 'unicode_escape')
    data.drop(data.index[50000:], 0, inplace=True)
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

    data['category'] = data["INJ_BODY_PART"].apply(lambda x: str(x).split('.')[0])

    data = data.rename(columns={'NARRATIVE':'text'})
    data['text_clean'] = data['text'].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))

    data_new = data
    data_new['text_clean'] = data['text_clean']
    data_new['INJ_BODY_PART'] =   data['INJ_BODY_PART']
    print("Here it is", data_new['INJ_BODY_PART'])
    X_train, X_test, y_train, y_test = train_test_split(data['text_clean'], data['INJ_BODY_PART'], test_size=0.1, random_state=1773)


## END OF PREPROCESSING

categories = ['HAND','KNEE','EYE','ANKLE','SHOULDER','BACK','OTHER']


if flag == "train":
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    # pickle.dump(vectorizer, open("vectorizerRF.pickle", "wb")) #Save vectorizer
    pickle.dump(vectorizer, open("vectorizerTFIDFRF.pickle", "wb")) #Save vectorizer
vectorizer = pickle.load(open("vectorizerTFIDFRF.pickle", 'rb'))      #Load vectorizer
# vectorizer = pickle.load(open("vectorizerRF.pickle", 'rb'))      #Load vectorizer
X_test = vectorizer.transform(X_test)


# For random forest


print("Random forest results")
filename = 'rfModel.sav'
if flag == "train":
    # The commented code is used for Hyper parameter tuning
    # param_grid = {'bootstrap': [True, False],
    # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    # 'max_features': ['auto', 'sqrt'],
    # 'min_samples_leaf': [1, 2, 4],
    # 'min_samples_split': [2, 5, 10],
    # 'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    # }
    # rf = RandomForestClassifier()
    # grid = GridSearchCV(rf, param_grid, verbose = 3, cv=10) 
    # # grid.fit(X_train, y_train)     # fitting the model for grid search
    # # print(grid.best_params_)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=10, scoring='f1_micro')
    rf.fit(X_train, y_train)
    pickle.dump(rf, open(filename, 'wb'))
    print(scores)
loaded_model_rf = pickle.load(open(filename, 'rb'))
y_pred = loaded_model_rf.predict(X_test)
labels = ['ANKLE','BACK','EYE','HAND','KNEE','OTHER','SHOULDER']   
probs = loaded_model_rf.predict_proba(X_test)
k = 4
best_n = np.argsort(-probs, axis=1)[:, :k]
file_name = 'rf_output.txt'

res = []
with open(file_name, 'w') as f:      
    for i in range(0, best_n.shape[0]):
        s = ''
        for j in range(0,k):
            f.write(labels[best_n.item((i,j))])
            s = s + labels[best_n.item((i,j))] + '.'
            f.write(".")
        f.write("\n")
        res.append(s)
for i in range(0, X_test.get_shape()[0]):
    results = loaded_model_rf.predict_proba(X_test)[i]
    prob_per_class_dictionary = dict(zip(loaded_model_rf.classes_, results))

if flag == "train":
    report = classification_report(y_test, y_pred, output_dict = True)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('rfk1.csv') 
    print(confusion_matrix(y_test, y_pred))



if flag == "test":
    for GROUND_TRUTH in GROUND_TRUTH_LIST:
        data['category' + GROUND_TRUTH] = data[GROUND_TRUTH].apply(lambda x: str(x).split('.')[0])
        
        for index, row in data.iterrows():
            y_test_labels[GROUND_TRUTH].append(str(row['category' + GROUND_TRUTH]))
        print(len(y_test_labels[GROUND_TRUTH]), len(y_pred))
        report = classification_report(y_test_labels[GROUND_TRUTH], y_pred, output_dict = True)
        df = pd.DataFrame(report).transpose()
        print(df)
        df.to_csv('rfk1' + GROUND_TRUTH +  '.csv') 
    
    final_data = pd.read_csv('rfK4.csv')
    final_data['preds'] = res
    final_data.to_csv('predictionsRF.csv')
    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    classes = ['ankle', 'back', 'eye', 'hand', 'knee', 'other', 'shoulder']
    oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsRF.csv', 'reedLabels', 'preds') # RF
    print("Random Forest - Reed ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    df.to_csv('rf_result_K4_reed.csv')
    print(df)
    print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))

    Y_TRUE = []
    Y_PRED = []
    oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsRF.csv', 'brownLabels', 'preds') # RF
    print("Random Forest - Brown ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    df.to_csv('rf_result_K4_brown.csv')
    print(df)
    print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))

    Y_TRUE = []
    Y_PRED = []
    oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsRF.csv', 'glennaLabels', 'preds') # RF
    print("Random Forest - Glenna ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    df.to_csv('rf_result_K4_glenna.csv')
    print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))

    Y_TRUE = []
    Y_PRED = []
    oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsRF.csv', 'glennaLabels', 'brownLabels') # RF
    print("Random Forest - Glenna v brown ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    df.to_csv('rf_result_K4_brown_v_glenna.csv')
    print(df)
    print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))

    Y_TRUE = []
    Y_PRED = []
    oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsRF.csv', 'glennaLabels', 'reedLabels') # RF
    print("Random Forest - Glenna  v reed ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    df.to_csv('rf_result_K4_glenna_v_reed.csv')
    print(df)
    print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))

    Y_TRUE = []
    Y_PRED = []
    oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsRF.csv', 'brownLabels', 'reedLabels') # RF
    print("Random Forest - Brown  v reed ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    df.to_csv('rf_result_K4_brown_v_reed.csv')
    print(df)
    print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))



