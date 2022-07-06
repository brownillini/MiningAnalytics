## for data
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
## for processing
import re
import nltk
# nltk.download()
## for plotting
# import matplotlib.pyplot as plt
# import seaborn as sns
## for w2v
# import gensim
# import gensim.downloader as gensim_api
## for bert
import transformers

from torch import nn
from transformers import BertModel
from transformers import BertTokenizer
# from transformers.modeling_bert import BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
data = pd.read_csv('interaction.labeled.csv', encoding = 'unicode_escape')
print(data.head())
data = data.rename(columns={'DESC':'text'})
data.sample(5)

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   
    ##characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()    
    ## remove Stopwords
    # if lst_stopwords is not None:
    #     lst_text = [word for word in lst_text if word not in 
    #                 lst_stopwords]
                
    # ## Stemming (remove -ing, -ly, ...)
    # if flg_stemm == True:
    #     ps = nltk.stem.porter.PorterStemmer()
    #     lst_text = [ps.stem(word) for word in lst_text]
                
    # ## Lemmatisation (convert the word into root word)
    # if flg_lemm == True:
    #     lem = nltk.stem.wordnet.WordNetLemmatizer()
    #     lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

# lst_stopwords = nltk.corpus.stopwords.words("english")
# lst_stopwords

data['text_clean'] = data['text'].apply(lambda x: 
          utils_preprocess_text(x))
print(data.head())

## END OF PREPROCESSING

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


data['category'] = data['Label (L Brown)'].apply(lambda x: x.split('.')[0])
y_test = []
for index, row in data.iterrows():
    y_test.append(str(row['category']))
# print(y_test)
y_pred = []
print(data.head())


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

# np.random.seed(112)
# df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42), 
#                                      [int(.8*len(data)), int(.9*len(data))])
df_test = data
# print(len(df_train),len(df_val), len(df_test))

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


from torch.optim import Adam
from tqdm import tqdm

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
                  
EPOCHS = 5
model = BertClassifier()
LR = 1e-6
model.load_state_dict(torch.load('bert_model_2', map_location=torch.device('cpu')))  

# uncomment the line below if you wish to train the model
# train(model, df_train, df_val, LR, EPOCHS)
torch.save(model.state_dict(), 'bert_model')

def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
            #   print(output, " Here is output")
              m = torch.nn.Softmax(dim=1)
            #   print(test.get_batch_texts(101), m(output))
              tf_predictions = m(output)
              with open('outputnew.txt', 'a+') as f:
                  tf_predictions = tf_predictions.numpy()
                #   print(tf_predictions)
                #   print(type(tf_predictions))
                  top4 = np.argpartition(tf_predictions.ravel(), -4)[:4]
                  for label in top4:
                    print(label)
                    f.write(reverse_labels.get(int(label)))
                    f.write(".")
                  f.write("\n")
                    

              acc = (output.argmax(dim=1) == test_label).sum().item()
              y_pred.append(reverse_labels.get(int(output.argmax(dim=1))))
            #   print(y_pred)
              total_acc_test += acc
    f.close()
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    report = classification_report(y_test, y_pred, output_dict = True)
    df = pd.DataFrame(report).transpose()
    print(df)
    df.to_csv('bertbk1.csv')
    print(confusion_matrix(y_test, y_pred))
evaluate(model, df_test)