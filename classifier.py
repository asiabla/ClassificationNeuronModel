import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"

names = ["Recency (months)","Frequency (times)","Monetary (c.c. blood)","Time (months)","whether he/she donated blood in March 2007"]

blood_df = pd.read_csv(url, names=names) #load CVS data

#load Pandas Dataframe into numpy arrays
data = blood_df.loc[1:,["Recency (months)","Frequency (times)","Monetary (c.c. blood)","Time (months)"]].values.astype(np.int)
target = blood_df.loc[1 :,['whether he/she donated blood in March 2007']].values.astype(np.int)

#standarise data
data = StandardScaler().fit_transform(data)
data_train, data_test, target_train, target_test = \
train_test_split(data, target, test_size=0.3, random_state=545)

#data describe
print(blood_df.describe())
print(blood_df.info())
print("blood_train_data.shape", data_train.shape)
print("blood_test_data.shape", data_test.shape)
print("blood_target_train.shape", target_train.shape)
print("blood_target_test.shape", target_test.shape)

#implementation of Perceptron

ppn = Perceptron (max_iter = 200, eta0 = 0.1, random_state = 0, tol = None)
ppn.fit(data_train, target_train.ravel())
y_pred = ppn.predict(data_test)
print('Incorrectly classified samples: %d' %(target_test.ravel() != y_pred).sum())

from sklearn.metrics import accuracy_score

print('Accurancy: %f '%accuracy_score(target_test.ravel(), y_pred))


#implementation
neural_network = MLPClassifier(hidden_layer_sizes=(50,100),activation = 'relu', solver = 'adam', random_state=1)
neural_network.fit(data_train, target_train.ravel())
pr = neural_network.predict(data_test)


conf_matrix_neural_network = confusion_matrix(target_test.ravel(),pr)
print("Confusion_matrix:")
print(conf_matrix_neural_network)
print("Number of connection between input and first hidden layer:")
print(np.size(neural_network.coefs_[0]))

print("Number of connection between first and second hidden layer:")
print(np.size(neural_network.coefs_[1]))
acc = accuracy_score(target_test.ravel(),pr)
print("Neural network model accuracy is {0:0.2f}".format(acc))

