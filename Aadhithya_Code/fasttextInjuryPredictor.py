## libraries to import
import fasttext
import nltk
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import average_precision

##### CHANGE THESE VARIABLES TO GET THE RESULTS FOR TWO SEPARATE GROUND TRUTHS. PLEASE UNCOMMENT THE GROUND TRUTH YOU WANT TO CHECK OUT.


GROUND_TRUTH = 'Label (R Reed)'
# GROUND_TRUTH = 'Label (L Brown)'
# GROUND_TRUTH = 'Label (Glenna)'


# flag = "train"
flag = "test"

if GROUND_TRUTH == 'Label (L Brown)':
    file_name = 'interactionFastTextBrown.txt'
    output_file = 'fasttext_output_brown.txt'
elif GROUND_TRUTH == 'Label (R Reed)':
    file_name = 'interactionFastTextReed.txt'
    output_file = 'fasttext_output_reed.txt'
elif GROUND_TRUTH == 'Label (Glenna)':
    file_name = 'interactionFastTextGlenna.txt' 
    output_file = 'fasttext_output_glenna.txt'
else:
    file_name = 'fastTextInjuryBig.preprocessed.txt'
    output_file = 'fasttext_output.txt'



if flag =="train":
    model = fasttext.train_supervised(input='fasttextInjurySmall.train', autotuneValidationFile='fasttextInjurySmall.valid', autotuneDuration=150)
    model.save_model("fasttext_model_small.bin")
model = fasttext.load_model("fasttext_model_small.bin")
y_pred = []
y_test = [] 

print(model.get_labels())
data = pd.read_csv('interaction.labeled.csv', encoding = 'unicode_escape')
data = data.rename(columns={'DESC':'text'})
data['category'] = data[GROUND_TRUTH].apply(lambda x: str(x).split('.')[0].lower())
print(data.head())
data["newtext"] = "__label__" + data["category"] + " " + data["text"]
with open(file_name, 'w') as f:
    for index, row in data.iterrows():
        f.write(str(row["newtext"]))
        f.write("\n")
f.close()
final_data = pd.read_csv('fasttextK4.csv')
res = []


with open(file_name, 'r') as f:
    lines = [line.rstrip() for line in f]
with open(output_file, 'w') as g:
    for line in lines:
        label = line.split(' ')[0]
        tup = (label,)
        y_test.append(tup)
        results = model.predict(line, k=4)
        g.write(str(results[0]))
        res.append(str(results[0]))
        g.write("\n")
        y_pred.append(model.predict(line)[0])
final_data["preds"] = res
final_data.to_csv("predictionsFT.csv")
print(model.test(file_name))






report = classification_report(y_test, y_pred, output_dict = True)
df = pd.DataFrame(report).transpose()

if GROUND_TRUTH == 'Label (L Brown)':
    df.to_csv('ftbk1.csv')
elif GROUND_TRUTH == 'Label (R Reed)':
    df.to_csv('ftrk1.csv') 
elif GROUND_TRUTH == 'Label (Glenna)':
    df.to_csv('ftgk1.csv') 
else:
    df.to_csv('ftk1.csv') 
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("MSHA dataset accuracy")
with open('fastTextInjuryBig.preprocessed.txt', 'r') as f:
        lines = [line.rstrip() for line in f]
with open('fasttext_output.txt', 'w') as g:
    for line in lines[110000:113951]:
        label = line.split(' ')[0]
        tup = (label,)
        y_test.append(tup)
        results = model.predict(line)
        g.write(str(results[0]))
        g.write("\n")
        y_pred.append(model.predict(line)[0])
report  = classification_report(y_test, y_pred, output_dict = True)
print(classification_report(y_test, y_pred))
df = pd.DataFrame(report).transpose()
df.to_csv('fasttexttMSHA.csv') 

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=False, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   
    ##characters and then strip)
    # text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = str(text).lower()
    ## Tokenize (convert from string to list)
    lst_text = text.split()    
    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    # back to string from list
    # text = " ".join(lst_text)
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
Y_TRUE = []
Y_PRED = []
# For K = 4 
classes = ['ankle', 'back', 'eye', 'hand', 'knee', 'other', 'shoulder']
oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsFT.csv', 'reedLabels', 'preds') # SVM
print("Fast text - Reed ground truth")
report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
df = pd.DataFrame(report).transpose()
df.to_csv('fasttext_result_K4_reed.csv')
print(df)
print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))

Y_TRUE = []
Y_PRED = []
oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsFT.csv', 'brownLabels', 'preds') # SVM
print("Fast text - Brown ground truth")
report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
df = pd.DataFrame(report).transpose()
df.to_csv('fasttext_result_K4_brown.csv')
print(df)
print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))

Y_TRUE = []
Y_PRED = []
oneHotEncoding(Y_TRUE, Y_PRED, 'predictionsFT.csv', 'glennaLabels', 'preds') # SVM
print("Fast text - Glenna ground truth")
report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
df = pd.DataFrame(report).transpose()
df.to_csv('fasttext_result_K4_glenna.csv')
print(df)
print(average_precision.mapk(Y_TRUE, Y_PRED, k =4))
# preprocessing to append labels with __label__ This section is used to create the fastTextInjuryBig.txt file.
def preprocess():
    data = pd.read_csv('MSHA.injuries.csv', encoding = 'unicode_escape')

    for index, row in data.iterrows():
        if row["INJ_BODY_PART"] == "FINGER(S)/THUMB":
           row["INJ_BODY_PART"] = """__label__HAND"""
        if row["INJ_BODY_PART"] =="""EYE(S) OPTIC NERVE/VISON""":
            row["INJ_BODY_PART"] = """__label__EYE"""
        if row["INJ_BODY_PART"] =="""HAND (NOT WRIST OR FINGERS)""" :
            row["INJ_BODY_PART"] = """__label__HAND"""
        if row["INJ_BODY_PART"] =="""FINGER(S)/THUMB""" :
            row["INJ_BODY_PART"] = """__label__HAND"""
        if row["INJ_BODY_PART"] =="""WRIST""" :
            row["INJ_BODY_PART"] = """__label__HAND"""
        if row["INJ_BODY_PART"] =="""ANKLE""" :
            row["INJ_BODY_PART"] =  """__label__ANKLE"""
        if row["INJ_BODY_PART"] =="""KNEE/PATELLA""" :
            row["INJ_BODY_PART"] = """__label__KNEE"""
        if row["INJ_BODY_PART"] =="""SHOULDERS (COLLARBONE/CLAVICLE/SCAPULA)""" :
            row["INJ_BODY_PART"] = """__label__SHOULDER"""
        if row["INJ_BODY_PART"] =="""BACK (MUSCLES/SPINE/S-CORD/TAILBONE)""" :
            row["INJ_BODY_PART"] = """__label__BACK"""
        if row["INJ_BODY_PART"] =="""FOREARM/ULNAR/RADIUS""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """ABDOMEN/INTERNAL ORGANS""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """HIPS (PELVIS/ORGANS/KIDNEYS/BUTTOCKS)""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """ELBOW""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """FOOT(NOT ANKLE/TOE)/TARSUS/METATARSUS""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """MOUTH/LIP/TEETH/TONGUE/THROAT/TASTE""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """SCALP""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """CHEST (RIBS/BREAST BONE/CHEST ORGNS)""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """LOWER LEG/TIBIA/FIBULA""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """NECK""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """JAW INCLUDE CHIN""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """TOE(S)/PHALANGES""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """EAR(S) INTERNAL & HEARING""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """UPPER ARM/HUMERUS""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """BRAIN""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """THIGH/FEMUR"""  :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """NOSE/NASAL PASSAGES/SINUS/SMELL"""  :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """EAR(S) EXTERNAL""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """SKULL""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """EAR(S) INTERNAL & EXTERNAL""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """FACE,NEC""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """ARM,NEC""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] ==  """HEAD,NEC""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """LEG, NEC""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """TRUNK,NEC""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """BODY PARTS, NEC"""  :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """LOWER EXTREMITIES,NEC""" :
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """UPPER EXTREMITIES, NEC""":
            row["INJ_BODY_PART"] = """__label__OTHER"""
        if row["INJ_BODY_PART"] == """BODY SYSTEMS""":
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """MULTIPLE PARTS (MORE THAN ONE MAJOR)""":
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """TRUNK, MULTIPLE PARTS""" :
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """UPPER EXTREMITIES, MULTIPLE""":
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """LOWER EXTREMITIES, MULTIPLE PARTS""":
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """FACE, MULTIPLE PARTS""" :
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """ARM, MULTIPLE PARTS""" :
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """HEAD, MULTIPLE PARTS""":
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """LEG, MULTIPLE PARTS""" :
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """NO VALUE FOUND""" :
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""
        if row["INJ_BODY_PART"] == """UNCLASSIFIED""" :
            row["INJ_BODY_PART"] = """__label__EXCLUDE"""

    for index, row in data.iterrows():
        if row["INJ_BODY_PART"] == """__label__EXCLUDE""":
            data.drop(index, inplace=True)
    data = data.rename(columns={'NARRATIVE':'text'})
    data["newtext"] = data["INJ_BODY_PART"] + " " + data["text"]

    for index, row in data.iterrows():
        with open('fastTextInjuryBig.txt', 'a+') as f:
                f.write(str(row["newtext"]))
                f.write("\n")


# preprocess() # uncomment this line to create the fastTextInjuryBig.txt file if need be