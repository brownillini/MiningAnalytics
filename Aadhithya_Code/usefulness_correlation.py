import pandas as pd

data = pd.read_csv('Additional_Interest_Measures_For_Useful_Rules (1).csv', encoding = 'unicode_escape')
data['Expert Comments'] = data['Expert Comments'].fillna(0)
print(data)
correlation_df = data.corrwith(data['Expert Usefulness Score'])
print(correlation_df)