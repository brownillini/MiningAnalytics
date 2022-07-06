from audioop import avg
import csv
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

Y_TRUE = {'HAND':[],'KNEE':[],'EYE':[],'ANKLE':[],'SHOULDER':[],'BACK':[],'OTHER':[]}
Y_SCORES = {'HAND':[],'KNEE':[],'EYE':[],'ANKLE':[],'SHOULDER':[],'BACK':[],'OTHER':[]}
def apk(index, bodyPart, comparisonIndex, groundTruthIndex):
    with open('interaction.labeled.bert.values.csv', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        y_true = np.array([])
        y_scores = np.array([]) 
        for row in reader:
            if row[groundTruthIndex].find(bodyPart) != -1:
                y_true = np.append(y_true, [1])
            else:
                y_true = np.append(y_true, [0])
            if float(row[comparisonIndex]) >= 0.14:
                y_scores = np.append(y_scores, [1])
            else:
                y_scores = np.append(y_scores, [0])

        #Precision Calculation
        precision = precision_score(y_true, y_scores)
        
        
        avg_precision = average_precision_score(y_true, y_scores)


        #Recall Calculation
        recall = recall_score(y_true, y_scores)

        x = y_true.tolist()
        y = y_scores.tolist()
        # prec, rec, _ = precision_recall_curve(x,y)
        # PrecisionRecallDisplay.from_predictions(x,y)
        precision, recall, _ = precision_recall_curve(x, y)
        # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot() 
        
        Y_TRUE[bodyPart].append(x)
        Y_SCORES[bodyPart].append(y)
        # print("Precision:" ,bodyPart, precision)
        # print("Average precision:", bodyPart, avg_precision)
        # print("Recall:" ,bodyPart, recall)

    return (precision, avg_precision, recall)
    
# Hand knee eye ankle shoulder back other
def calculatePR(bodyPart, comparisonIndex, groundTruthIndex):
    with open('interaction.labeled.bert.values.csv', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        y_true = np.array([])
        y_scores = np.array([]) 
        for row in reader:
            if row[groundTruthIndex].find(bodyPart) != -1:
                y_true = np.append(y_true, [1])
            else:
                y_true = np.append(y_true, [0])
            if float(row[comparisonIndex]) >= 0.14:
                y_scores = np.append(y_scores, [1])
            else:
                y_scores = np.append(y_scores, [0])

        #Precision Calculation
        precision = precision_score(y_true, y_scores)
        
        avg_precision = average_precision_score(y_true, y_scores)


        #Recall Calculation
        recall = recall_score(y_true, y_scores)

        # print("Precision:" ,bodyPart, precision)
        # print("Average precision:", bodyPart, avg_precision)
        # print("Recall:" ,bodyPart, recall)

    return (precision, recall)


MachinevBrownPRTuple = []
MachinevReedPRTuple = []
BrownvReedPRTuple = []
ReedvBrownPRTuple = []


mapk = []

# mapk.append(apk(0,'HAND', 4, 3))
# mapk.append(apk(1,'KNEE', 4, 3))
# mapk.append(apk(2,'EYE', 4, 3))
# mapk.append(apk(3,'ANKLE', 4, 3))
# mapk.append(apk(4,'SHOULDER', 4, 3))
# mapk.append(apk(5,'BACK', 4, 3))
# mapk.append(apk(6,'OTHER', 4, 3))
# print(Y_TRUE)
# print(Y_SCORES)
precision = dict()
recall = dict()
average_precision = dict()
categories = ['HAND','KNEE','EYE','ANKLE','SHOULDER','BACK','OTHER']

# sum = 0.0
# for i in mapk:
#     sum += i[1]
# print("MAPK  machine v reed = ", sum/7)

# mapk.append(apk(0,"HAND", 7, 9))
# mapk.append(apk(1,"KNEE", 7, 9))
# mapk.append(apk(2,"EYE", 7, 9))
# mapk.append(apk(3,"ANKLE", 7, 9))
# mapk.append(apk(4,"SHOULDER", 7, 9))
# mapk.append(apk(5,"BACK", 7, 9))
# mapk.append(apk(6,"OTHER", 7, 9))
# sum = 0.0
# for i in mapk:
#     sum += i[1]
# print("MAPK  brown v reed = ", sum/7)

# Dr.Brown's values as ground truth
MachinevBrownPRTuple.append(calculatePR("HAND",12, 10))
MachinevBrownPRTuple.append(calculatePR("KNEE",13, 10))
MachinevBrownPRTuple.append(calculatePR("EYE",14, 10))
MachinevBrownPRTuple.append(calculatePR("ANKLE",15, 10))
MachinevBrownPRTuple.append(calculatePR("SHOULDER",16, 10))
MachinevBrownPRTuple.append(calculatePR("BACK",17, 10))
MachinevBrownPRTuple.append(calculatePR("OTHER",18, 10))
# print(MachinevBrownPRTuple)
# mapk.append(apk(0,"HAND", 4, 2))
# mapk.append(apk(1,"KNEE", 4, 2))
# mapk.append(apk(2,"EYE", 4, 2))
# mapk.append(apk(3,"ANKLE", 4, 2))
# mapk.append(apk(4,"SHOULDER", 4, 2))
# mapk.append(apk(5,"BACK", 4, 2))
# mapk.append(apk(6,"OTHER", 4, 2))
# sum = 0.0
# for i in mapk:
#     sum += i[1]
# print("MAPK  machine v brown = ", sum/7)

sumP = 0
sumR = 0
for i in MachinevBrownPRTuple:
    sumP += i[0]
    sumR += i[1]
# print(float(sumP)/7, float(sumR)/7)
MachinevBrownAvgP = float(sumP)/7
MachinevBrownAvgR = float(sumR)/7

# ReedvBrownPRTuple.append(calculatePR("HAND", 9, 7))
# ReedvBrownPRTuple.append(calculatePR("KNEE", 9, 7))
# ReedvBrownPRTuple.append(calculatePR("EYE", 9, 7))
# ReedvBrownPRTuple.append(calculatePR("ANKLE", 9, 7))
# ReedvBrownPRTuple.append(calculatePR("SHOULDER", 9, 7))
# ReedvBrownPRTuple.append(calculatePR("BACK", 9, 7))
# # ReedvBrownPRTuple.append(calculatePR("OTHER", 9, 7))
# # print(ReedvBrownPRTuple)

# sumP = 0
# sumR = 0
# for i in ReedvBrownPRTuple:
#     sumP += i[0]
#     sumR += i[1]
# # print(float(sumP)/7, float(sumR)/7)
# ReedvBrownAvgP = float(sumP)/7
# ReedvBrownAvgR = float(sumR)/7


# Dr.Reed's values as ground truth
MachinevReedPRTuple.append(calculatePR("HAND", 12, 8))
MachinevReedPRTuple.append(calculatePR("KNEE", 13, 8))
MachinevReedPRTuple.append(calculatePR("EYE", 14, 8))
MachinevReedPRTuple.append(calculatePR("ANKLE", 15, 8))
MachinevReedPRTuple.append(calculatePR("SHOULDER", 16, 8))
MachinevReedPRTuple.append(calculatePR("BACK", 17, 8))
MachinevReedPRTuple.append(calculatePR("OTHER", 18, 8))
# print(MachinevReedPRTuple)

sumP = 0
sumR = 0
for i in MachinevReedPRTuple:
    sumP += i[0]
    sumR += i[1]
# print(float(sumP)/7, float(sumR)/7)
MachinevReedAvgP = float(sumP)/7
MachinevReedAvgR = float(sumR)/7



# BrownvReedPRTuple.append(calculatePR("HAND", 9, 7))
# BrownvReedPRTuple.append(calculatePR("KNEE", 9, 7))
# BrownvReedPRTuple.append(calculatePR("EYE", 9, 7))
# BrownvReedPRTuple.append(calculatePR("ANKLE", 9, 7))
# BrownvReedPRTuple.append(calculatePR("SHOULDER", 9, 7))
# BrownvReedPRTuple.append(calculatePR("BACK", 9, 7))
# BrownvReedPRTuple.append(calculatePR("OTHER", 9, 7))
# # print(BrownvReedPRTuple)

# sumP = 0
# sumR = 0
# for i in BrownvReedPRTuple:
#     sumP += i[0]
#     sumR += i[1]
# # print(float(sumP)/7, float(sumR)/7)
# BrownvReedAvgP = float(sumP)/7
# BrownvReedAvgR = float(sumR)/7


# print("Brown v Reed(Ground Truth) Precision and Recall    :", BrownvReedAvgP, BrownvReedAvgR)
print("Machine v Reed(Ground Truth) Precision and Recall  :", MachinevReedAvgP, MachinevReedAvgR)
print("----------------------------------------------------")
# print("Reed v Brown(Ground Truth) Precision and Recall    :", ReedvBrownAvgP, ReedvBrownAvgR)
print("Machine v Brown(Ground Truth) Precision and Recall :", MachinevBrownAvgP, MachinevBrownAvgR)
print("----------------------------------------------------")

################################################################################
    # My defined function for precision recall
    #   tp = 0
    #   fp = 0
    #   fn = 0
    #   for row in reader:
    #       if row[comparisonIndex].find(bodyPart) != -1 and row[groundTruthIndex].find(bodyPart) != -1:
    #           tp+=1
    #       if row[comparisonIndex].find(bodyPart) != -1 and row[groundTruthIndex].find(bodyPart) == -1:
    #           fp +=1
    #       if row[comparisonIndex].find(bodyPart) == -1 and row[groundTruthIndex].find(bodyPart) != -1:
    #           fn +=1
    # precision = float(tp)/(tp + fp)
    # recall = float(tp)/(tp + fn)
    # print("Precision: ", bodyPart, str(precision))
    # print("Recall: ", bodyPart,  str(recall))