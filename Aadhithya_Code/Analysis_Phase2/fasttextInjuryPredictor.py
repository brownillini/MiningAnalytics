## libraries to import
import fasttext
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

##### CHANGE THESE VARIABLES TO GET THE RESULTS FOR TWO SEPARATE GROUND TRUTHS. PLEASE UNCOMMENT THE GROUND TRUTH YOU WANT TO CHECK OUT.

# flag = "train"
flag = "test"

if flag =="train":
    model = fasttext.train_supervised(input='risk_rating.train', autotuneValidationFile='risk_rating.valid')
    model.save_model("risk_rating_fasttext_model.bin")

model = fasttext.load_model("risk_rating_fasttext_model.bin")
model.test("risk_rating_test.valid")
#Uncomment the code below to find the best parameters

# args_obj = model.f.getArgs()
# for hparam in dir(args_obj):
#     if not hparam.startswith('__'):
#         print(f"{hparam} -> {getattr(args_obj, hparam)}")

y_pred = []
y_test = [] 

print(model.get_labels())



with open('risk_rating_test.txt', 'r') as f:
    lines = [line.rstrip() for line in f]

for line in lines:
    label = line.split(' ')[0]
    tup = (label,)
    y_test.append(tup)
    results = model.predict(line)
    y_pred.append(model.predict(line)[0])
report = classification_report(y_test, y_pred, output_dict = True)
# print(report)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# df = pd.DataFrame(report).transpose()

# df.to_csv('fast_text_results.csv')






