## libraries to import
import fasttext
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

##### CHANGE THESE VARIABLES TO GET THE RESULTS FOR TWO SEPARATE GROUND TRUTHS. PLEASE UNCOMMENT THE GROUND TRUTH YOU WANT TO CHECK OUT.

# Uncomment the below variables for Reed Ground Truth
# file_name = 'interactionFastTextReed.txt' # For Reed ground truth
# output_file = 'fasttext_output_reed.txt' # For Reed ground truth
# GROUND_TRUTH = 'Label (R Reed)'


# Uncomment the below variables for Brown Ground Truth
file_name = 'interactionFastText.txt' # For Brown ground truth
output_file = 'fasttext_output_brown.txt' # For Brown ground truth
GROUND_TRUTH = 'Label (L Brown)'

model = fasttext.train_supervised(input = "fastTextInjuryBig.txt", lr = 0.01, epoch = 20)
y_pred = []
y_test = [] 

print(model.get_labels())
data = pd.read_csv('interaction.labeled.csv', encoding = 'unicode_escape')
data = data.rename(columns={'DESC':'text'})
data['category'] = data[GROUND_TRUTH].apply(lambda x: x.split('.')[0])
print(data.head())
data["newtext"] = "__label__" + data["category"] + " " + data["text"]
with open(file_name, 'w') as f:
    for index, row in data.iterrows():
        f.write(str(row["newtext"]))
        f.write("\n")
f.close()

with open(file_name, 'r') as f:
    lines = [line.rstrip() for line in f]
with open(output_file, 'w') as g:
    for line in lines:
        label = line.split(' ')[0]
        tup = (label,)
        y_test.append(tup)
        results = model.predict(line, k=4)
        g.write(str(results[0]))
        g.write("\n")
        y_pred.append(model.predict(line)[0])
    
print(model.test(file_name))
# print(y_pred)
# print(y_test)

report = classification_report(y_test, y_pred, output_dict = True)
df = pd.DataFrame(report).transpose()
if GROUND_TRUTH == 'Label (L Brown)':
    df.to_csv('ftbk1.csv')
elif GROUND_TRUTH == 'Label (R Reed)':
    df.to_csv('ftrk1.csv') 
else:
    df.to_csv('ftk1.csv') 
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))




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