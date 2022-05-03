#Menna Hamdy Mahmoud 20190558


import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

weather_df = pd.read_csv('weatherHistory.csv')
weather_df.isna().sum()
weather_df.dropna(inplace=True)
weather_df.shape
weather_df.head()

x = weather_df.drop("Humidity", axis=1)
y = weather_df['Humidity']

x.drop(['Daily Summary','Formatted Date'],axis=1,inplace=True)

x=pd.get_dummies(x,columns=['Summary','Precip Type'],drop_first=True)

train_X,test_X,train_y,test_y = train_test_split(x,y,test_size = 0.3 ,random_state=10)
scaler = StandardScaler()

train_X=scaler.fit_transform(train_X)
test_X=scaler.transform(test_X)

class ADALINE(object):

    def __init__(self, it=50, random_state=1, rate=0.00001):
        self.it = it

        self.random_state = random_state

        self.rate = rate

    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """
        # weights
        rand = np.random.RandomState(self.random_state)
        self.weight = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        # Number of misclassifications
        self.errors = []

        for i in range(self.it):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()

    # return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return self.activation(X)

meanSqauredErrors = []
for i in range(50):
    model = ADALINE(it=i)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    meanSqauredErrors.append(mean_squared_error(test_y, y_pred))

plt.plot(meanSqauredErrors)
plt.xlabel('Number of iterators')
plt.ylabel('Mean Squared Error')
plt.title('ADALINE Project')

# the MSE was low tell the iterator = 30 after that MSE raised up