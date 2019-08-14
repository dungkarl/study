import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_dict = {
    'a':[1,2],
    'b':[3,4]
}

df = pd.DataFrame(data_dict)
print(df)
sheet_name = 'Sheet1'
file_name = 'data_demo_1k.xlsx'

with open(file_name, 'rb') as file:
    data_demo = pd.read_excel(file, sheetname = sheet_name)
# print(data_demo[:4])
# print(data_demo.columns)
print(data_demo.head())

# header = data_demo['STT']
# print(header)
df.to_excel('test.xlsx', index=False)
# print(df.loc[0])
# plt.plot(df)
# plt.show()
my_head = df.head()
print(my_head)
# title = df.title()
# print(title)
