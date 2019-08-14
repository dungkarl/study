import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.style.use('ggplot')

x = [1, 3,4,2,10,22,3,4,5,6,7,8,18]
df = pd.DataFrame(x, columns=['Test'])
print(df)
data_field = df.values
min_max_scaler = preprocessing.MinMaxScaler()
data_field_scaled = min_max_scaler.fit_transform(data_field)
new_df = pd.DataFrame(data_field_scaled)
#print(df.where(df.values > 10))
print(new_df)
#plt.plot(data_field)
plt.plot(data_field_scaled)
plt.show()
