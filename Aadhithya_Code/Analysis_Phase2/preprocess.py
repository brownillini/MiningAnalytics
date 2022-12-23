import pandas as pd
import numpy as np
import fasttext

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
# import average_precision

from deep_translator import GoogleTranslator
translated = GoogleTranslator(source='auto', target='de').translate("keep it up, you are awesome")

data = pd.read_csv("""Consolidated_2021_incidents.csv""", encoding= 'utf-8')


# for index, text in enumerate(data['Incident Description']):
    
#     x = nltk.tokenize.sent_tokenize(str(text))
#     translated_text = ''
#     for sentence in x:
#         text = GoogleTranslator(source='auto', target='en').translate(sentence)
#         if text:
#             translated_text += text
#     text = translated_text
#     data['Incident Description'][index] = text
#     print(text)
    

# data.to_csv("Translated_Incidents_2021.csv")

df1 = pd.read_csv("""Translated_Incidents_2021.csv""", encoding= 'utf-8')
df2 = pd.read_csv("""Translated_Incidents_2022.csv""", encoding= 'utf-8')

df = pd.concat([df1,df2]).drop_duplicates().reset_index(drop=True)

for col in df.columns:
    print(col)
incidentNumbers = set()

# df.to_csv("translated_removed_duplicates.csv")

# for index, row in df.iterrows():
#     if row["Near Miss?"] == "Yes" and row["Incident No."] not in incidentNumbers:
#         incidentNumbers.add(row["Incident No."])
#     else:
#         df.drop(index, inplace=True)
# final_df = df.sort_values(by=['Unnamed: 0'])
# final_df.to_csv("Near_misses.csv")




    # if row["Near Miss?"] == "Yes" and row["Incident No."] not in incidentNumbers:
    #     if str(row["Incident Type"]).find("Equipment") != -1 and str(row["Incident Type"]).find("Environment" ) != -1:
    #         incidentNumbers.add(row["Incident No."])   
    #     elif row["Hazard?"] == "Yes":
    #         incidentNumbers.add(row["Incident No."])
    #     else:
    #         df.drop(index, inplace=True)             
    # else:
    #     df.drop(index, inplace=True)


# print(list(data))
# print(data['Incident Description'])




df = df.rename(columns={'Incident Description':'text'})
df["newtext"] = "__label__" + df["Risk Rating"] + " " + df["text"]

for index, row in df.iterrows():
    with open('risk_rating_input.txt', 'a+') as f:
            f.write(str(row["newtext"]))
            f.write("\n")
