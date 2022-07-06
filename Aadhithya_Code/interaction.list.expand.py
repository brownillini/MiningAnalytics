import pandas as pd

data = pd.read_csv('interaction.labeled.csv', encoding = 'unicode_escape')
data = data.rename(columns={'DESC':'text'})
data['category'] = data['Label (L Brown)'].apply(lambda x: x.split('.'))

for index, row in data.iterrows():
    for label in row['category']:
        newtext = "__label__" + label
        with open('interaction.list.expanded.brown.new.txt', 'a+') as f:
            f.write(str(newtext))
            f.write("\n")
