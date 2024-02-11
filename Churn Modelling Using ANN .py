# Artificial Neural Network
import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing

# Importing the dataset
dataset=pd.read_csv(r"D:\DL Projects\Churn Modelling-ANN\Churn Modelling Using MNN or DNN\Churn_Modelling.csv")
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x, y,test_size=0.2,random_state=0)

# Part 2 - Building the ANN

# Initializing the ANN
from tensorflow import keras
ann = keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

# Adding the third hidden layer
ann.add(tf.keras.layers.Dense(units=5,activation='relu'))

# Adding the fourth hidden layer
ann.add(tf.keras.layers.Dense(units=4,activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# Training the ANN on the Training set
ann.fit(xtrain,ytrain,batch_size=32,epochs=150)

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
ypred=ann.predict(xtest)
ypred=(ypred>0.5)
np.concatenate((ypred.reshape(len(ypred),1),ytest.reshape(len(ytest),1)),1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(ytest,ypred)
print(ac)
