from turtle import pd
import pandas as pd
data = pd.read_csv("""MSHA.injuries.csv""", encoding= 'unicode_escape')
print(data.head())
data.drop(data.index[50000:], 0, inplace=True)

# data.replace( to_replace="""EYE(S) OPTIC NERVE/VISON""",value="""EYE""")
# data.replace(to_replace="""HAND (NOT WRIST OR FINGERS)""" ,value= """HAND""")
# data.replace(to_replace="""FINGER(S)/THUMB""" ,value= """HAND""")
# data.replace(to_replace="""WRIST""" ,value= """HAND""")
# data.replace(to_replace="""ANKLE""" ,value=  """ANKLE""")
# data.replace(to_replace="""KNEE/PATELLA""" ,value= """KNEE""")
# data.replace(to_replace="""SHOULDERS (COLLARBONE/CLAVICLE/SCAPULA)""" ,value= """SHOULDER""")
# data.replace(to_replace="""BACK (MUSCLES/SPINE/S-CORD/TAILBONE)""" ,value= """BACK""")
# data.replace(to_replace="""FOREARM/ULNAR/RADIUS""" ,value= """OTHER""")
# data.replace(to_replace= """ABDOMEN/INTERNAL ORGANS""",value= """OTHER""")
# data.replace(to_replace= """HIPS (PELVIS/ORGANS/KIDNEYS/BUTTOCKS)""",value= """OTHER""")
# data.replace(to_replace= """ELBOW""" ,value= """OTHER""")
# data.replace(to_replace= """FOOT(NOT ANKLE/TOE)/TARSUS/METATARSUS""",value= """OTHER""")
# data.replace(to_replace= """MOUTH/LIP/TEETH/TONGUE/THROAT/TASTE""",value= """OTHER""")
# data.replace(to_replace= """SCALP""" ,value= """OTHER""")
# data.replace(to_replace= """CHEST (RIBS/BREAST BONE/CHEST ORGNS)""",value= """OTHER""")
# data.replace(to_replace= """LOWER LEG/TIBIA/FIBULA""",value= """OTHER""")
# data.replace(to_replace= """NECK""",value= """OTHER""")
# data.replace(to_replace= """JAW INCLUDE CHIN""" ,value= """OTHER""")
# data.replace(to_replace= """TOE(S)/PHALANGES""" ,value= """OTHER""")
# data.replace(to_replace= """EAR(S) INTERNAL & HEARING""" ,value= """OTHER""")
# data.replace(to_replace= """UPPER ARM/HUMERUS""",value= """OTHER""")
# data.replace(to_replace= """BRAIN""" ,value= """OTHER""")
# data.replace(to_replace= """THIGH/FEMUR"""  ,value= """OTHER""")
# data.replace(to_replace= """NOSE/NASAL PASSAGES/SINUS/SMELL"""  ,value= """OTHER""")
# data.replace(to_replace= """EAR(S) EXTERNAL""",value= """OTHER""")
# data.replace(to_replace= """SKULL""",value= """OTHER""")
# data.replace(to_replace= """EAR(S) INTERNAL & EXTERNAL""" ,value= """OTHER""")

# data.replace(to_replace= """BODY SYSTEMS""",value= """EXCLUDE""")
# data.replace(to_replace= """MULTIPLE PARTS (MORE THAN ONE MAJOR)""",value= """EXCLUDE""")
# data.replace(to_replace= """TRUNK, MULTIPLE PARTS""" ,value= """EXCLUDE""")
# data.replace(to_replace= """UPPER EXTREMITIES, MULTIPLE""",value= """EXCLUDE""")
# data.replace(to_replace= """LOWER EXTREMITIES, MULTIPLE PARTS""",value= """EXCLUDE""")
# data.replace(to_replace= """FACE, MULTIPLE PARTS""" ,value= """EXCLUDE""")
# data.replace(to_replace= """ARM, MULTIPLE PARTS""" ,value= """EXCLUDE""")
# data.replace(to_replace= """HEAD, MULTIPLE PARTS""",value= """EXCLUDE""")
# data.replace(to_replace= """LEG, MULTIPLE PARTS""" ,value= """EXCLUDE""")

# data.replace(to_replace= """FACE,NEC""" ,value= """OTHER""")
# data.replace(to_replace= """ARM,NEC""",value= """OTHER""")
# data.replace(to_replace=  """HEAD,NEC""",value= """OTHER""")
# data.replace(to_replace= """LEG, NEC""",value= """OTHER""")
# data.replace(to_replace= """TRUNK,NEC""",value= """OTHER""")
# data.replace(to_replace= """BODY PARTS, NEC"""  ,value= """OTHER""")
# data.replace(to_replace= """LOWER EXTREMITIES,NEC""" ,value= """OTHER""")
# data.replace(to_replace= """UPPER EXTREMITIES, NEC""",value= """OTHER""")

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

    if row["INJ_BODY_PART"] == """BODY SYSTEMS""":
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """MULTIPLE PARTS (MORE THAN ONE MAJOR)""":
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """TRUNK, MULTIPLE PARTS""" :
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """UPPER EXTREMITIES, MULTIPLE""":
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """LOWER EXTREMITIES, MULTIPLE PARTS""":
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """FACE, MULTIPLE PARTS""" :
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """ARM, MULTIPLE PARTS""" :
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """HEAD, MULTIPLE PARTS""":
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """LEG, MULTIPLE PARTS""" :
        data.drop(index, inplace=True)

    if row["INJ_BODY_PART"] == """NO VALUE FOUND""" :
        data.drop(index, inplace=True)


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
print(data.head())