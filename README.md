!pip install opendatasets  # to install opendataset python package to easily download the datsets from any platform



import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt

!pip install opendatasets

"""# New section"""

df = pd.read_csv("/content/Salary_Data.csv")
df.head()

sns.lmplot(x='YearsExperience',y='Salary',data=df)

model=linear_model.LinearRegression()

#df = df.dropna(subset=['y'])

model.fit(df[['YearsExperience']], df['Salary'])

model.score(df[['YearsExperience']], df['Salary'])

model.predict([[10]])

model.coef_

model.intercept_

y=10*9449.96232146+25792.200198668696

y

import numpy as np

class MereLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, x_train, y_train):
        # Ensure inputs are 1D numpy arrays
        x_train = np.array(x_train).flatten()
        y_train = np.array(y_train).flatten()

        num = 0
        den = 0
        x_mean = x_train.mean()
        y_mean = y_train.mean()

        for i in range(len(x_train)):
            num += (x_train[i] - x_mean) * (y_train[i] - y_mean)
            den += (x_train[i] - x_mean) * (x_train[i] - x_mean)

        self.m = num / den
        self.b = y_mean - (self.m * x_mean)

    def predict(self, x_test):
        x_test = np.array(x_test).flatten()
        return self.m * x_test + self.b

import numpy as np
import pandas as pd

df=pd.read_csv("/content/Salary_Data.csv")

df.head()



x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

x

y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

x_train.shape  ### we get the sample

lr = MereLR()  ##object of MereLR

lr.fit(x_train, y_train)

x_train.mean()

x_train.shape[0]

x_test.shape[0]

x_train[0]

y_train[0]

x_test[0]

y_test[0]

df



