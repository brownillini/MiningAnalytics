import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('SVG') #set the backend to SVG
from wordcloud import WordCloud

import re
import nltk


df = pd.read_csv("MSHA.injuries.csv", encoding= 'unicode_escape')

print(df.isna().sum())
df = df.rename(columns={'NARRATIVE':'text'})

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

lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords.append("ee")
lst_stopwords.append("employee")

text = " ".join(df['text'].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords)))
# print(text)

word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

fname = "cloud_test"
plt.imshow(word_cloud, interpolation="bilinear") 
plt.axis("off")
fig = plt.gcf() #get current figure
fig.set_size_inches(10,10)  
plt.savefig(fname, dpi=700)



# plt.figure( figsize=(20,10), facecolor='k')
# plt.imshow(word_cloud)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()



# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()