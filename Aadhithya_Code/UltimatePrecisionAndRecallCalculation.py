import csv
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)


# Hand knee eye ankle shoulder back other
def oneHotEncoding(Y_TRUE, Y_PRED, model, groundTruthIndex, comparisonIndex):
    df = pd.read_csv(model, encoding = 'unicode_escape')
    for index, row in df.iterrows():
        y_true = []
        y_scores = []
        classes = ['ANKLE', 'BACK', 'EYE', 'HAND', 'KNEE', 'OTHER', 'SHOULDER']
        for bodyPart in classes:
            if row[groundTruthIndex].find(bodyPart) != -1:
                y_true.append(1)
            if row[comparisonIndex].find(bodyPart) != -1:
                y_scores.append(1)
            if row[groundTruthIndex].find(bodyPart) == -1:
                y_true.append(0)
            if row[comparisonIndex].find(bodyPart) == -1:
                y_scores.append(0)
        Y_TRUE.append(y_true)
        Y_PRED.append(y_scores)


def main():
    classes = ['ANKLE', 'BACK', 'EYE', 'HAND', 'KNEE', 'OTHER', 'SHOULDER']
    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'svmK4.csv', 'reedLabels', 'preds') # SVM
    print("SVM - Reed ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('svmr.csv')
    
    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'svmK4.csv', 'brownLabels', 'preds') # SVM
    print("SVM - Brown ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('svmb.csv')

    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'rfK4.csv', 'reedLabels', 'preds') # Random Forest
    print("RF - Reed ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('rfr.csv')

    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'rfK4.csv', 'brownLabels', 'preds') # Random Forest
    print("RF - Brown ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('rfb.csv')

    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'fasttextK4.csv', 'reedLabels', 'preds') # Fast Text
    print("Fast Text - Reed ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('ftr.csv')

    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'fasttextK4.csv', 'brownLabels', 'preds') # Fast Text
    print("Fast Text - Brown ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('ftb.csv')

    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'bertK4.csv', 'reedLabels', 'preds') # BERT
    print("BERT - Reed ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('bertr.csv')

    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'bertK4.csv', 'brownLabels', 'preds') # BERT
    print("BERT - Brown ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('bertb.csv')

    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'bertK4.csv', 'reedLabels', 'brownLabels') # Reed Ground Truth
    print("Expert Comparison - Reed ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('rb.csv')

    Y_TRUE = []
    Y_PRED = []
    # For K = 4 
    oneHotEncoding(Y_TRUE, Y_PRED, 'bertK4.csv', 'brownLabels', 'reedLabels') # Brown Ground Truth
    print("Expert Comparison- Brown ground truth")
    report = classification_report(Y_TRUE, Y_PRED, output_dict = True, target_names=classes)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('br.csv')



if __name__ == "__main__":
    main()