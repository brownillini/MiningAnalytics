## libraries to import
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import re # for regex

import nltk # NLP toolkit
# nltk.download() # uncomment this when you run nltk for the first time
## for bert

from torch import nn

import transformers
from transformers import BertModel
from transformers import BertTokenizer

from torch.optim import Adam
from tqdm import tqdm
## end of libraries

##### CHANGE THESE VARIABLES TO GET THE RESULTS FOR TWO SEPARATE GROUND TRUTHS. PLEASE UNCOMMENT THE GROUND TRUTH YOU WANT TO CHECK OUT.

# GROUND_TRUTH = 'Label (L Brown)'
GROUND_TRUTH = 'Label (R Reed)'
# GROUND_TRUTH = 'MSHA' # this is set when training the model

flag = "test" # change the value to train if you wish to train bert
# flag = "train"

#### END OF VARIABLES

### Label definitions ###
# Labels used
labels = {'HAND':0,
          'KNEE':1,
          'EYE':2,
          'ANKLE':3,
          'SHOULDER':4,
          'BACK':5,
          'OTHER':6
          }

reverse_labels = {0:'HAND',
          1:'KNEE',
          2:'EYE',
          3:'ANKLE',
          4:'SHOULDER',
          5:'BACK',
          6:'OTHER'
          }

### End of label definitions

#### Class definitions ####

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text_clean']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 7) # there are 7 labels
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

### END OF CLASSES

### Functions Definitions ###

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=False, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   
    ##characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
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
    text = " ".join(lst_text)
    return text

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=7, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=7)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        print("|||Printing the predicted labels into a new file. Please wait for about 5 minutes.|||")
        if GROUND_TRUTH == 'Label (L Brown)':
            file_name = 'bert_output_brown.txt'
        elif GROUND_TRUTH == 'Label (R Reed)':
            file_name = 'bert_output_reed.txt'
        else:
            file_name = 'bert_output.txt'

        with open(file_name, 'w') as f:
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                m = torch.nn.Softmax(dim=1) # converting to probabilities
                tf_predictions = m(output)
                tf_predictions = tf_predictions.numpy()
                top4 = np.argpartition(tf_predictions.ravel(), -4)[:4] # getting the top 4 labels predicted by the model
                for label in top4:
                    f.write(reverse_labels.get(int(label)))
                    f.write(".")
                    print(reverse_labels.get(int(label)), end=".")
                f.write("\n")
                acc = (output.argmax(dim=1) == test_label).sum().item()
                y_pred.append(reverse_labels.get(int(output.argmax(dim=1))))
                total_acc_test += acc
    f.close()
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    print(len(y_test), len(y_pred))
    report = classification_report(y_test, y_pred, output_dict = True)
    df = pd.DataFrame(report).transpose()
    print(df)

    if GROUND_TRUTH == 'Label (L Brown)':
        df.to_csv('bertbk1.csv')
    elif GROUND_TRUTH == 'Label (R Reed)':
        df.to_csv('bertrk1.csv') 
    else:
        df.to_csv('bertk1.csv') 

    print(confusion_matrix(y_test, y_pred))

### END OF FUNCTIONS ###

### Script begins ###

tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # You may try out other pretrained tokenizers from https://huggingface.co/ 
## We are not removing the stopwords as BERT is a context-based model. You may experiment with the performance by removing the stopwords and performing lemmatisation/stemming
# lst_stopwords = nltk.corpus.stopwords.words("english")
# lst_stopwords

y_test = []
y_pred = []

if flag == "test":
    data = pd.read_csv('interaction.labeled.csv', encoding = 'unicode_escape')
    data = data.rename(columns={'DESC':'text'})
    data['text_clean'] = data['text'].apply(lambda x: 
          utils_preprocess_text(x))

    data['category'] = data[GROUND_TRUTH].apply(lambda x: x.split('.')[0])

    for index, row in data.iterrows():
        y_test.append(str(row['category']))
    df_test = data
else:
    np.random.seed(112)
    data = pd.read_csv("""MSHA.injuries.csv""", encoding= 'unicode_escape')
    data.drop(data.index[500:], 0, inplace=True)
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

    data['category'] = data["INJ_BODY_PART"].apply(lambda x: x.split('.')[0])


    data = data.rename(columns={'NARRATIVE':'text'})
    data['text_clean'] = data['text'].apply(lambda x: 
          utils_preprocess_text(x))
    df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42), 
                                            [int(.8*len(data)), int(.9*len(data))])
    for index, row in df_test.iterrows():
        y_test.append(str(row['category']))
    print(len(df_train),len(df_val), len(df_test))


## END OF PREPROCESSING



EPOCHS = 5
model = BertClassifier()
LR = 1e-6
model.load_state_dict(torch.load('bert_model_2', map_location=torch.device('cpu')))  


# train(model, df_train, df_val, LR, EPOCHS) # uncomment the line below if you wish to train the model
torch.save(model.state_dict(), 'bert_model')


# PLease wait for the entire labels
evaluate(model, df_test)


