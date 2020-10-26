import pandas as pd
import numpy as np

data_frame = pd.read_csv("D:\\Projects\\vkr\\assets\\data\\vehicles.csv")

# print(data_frame[['ASSET']].drop_duplicates().count())
print(type(data_frame['ASSET'] == 123))
