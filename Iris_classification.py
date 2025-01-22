import pandas as pd   #Library to handle dataframe
import matplotlib.pyplot as plt  # Library for visualization


df = pd.read_csv('Iris.csv')  #Load the dataset in a variable

print(df.head())  # display first 5 rows of dataset

X=df.iloc[:,1:-1] # Remove Id and Target column for prediction
y=df.iloc[:,-1] # Load the target column 

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,stratify=y)  #Split the dataset for training the model and evaluating


#Model training
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()  #Load the model

model.fit(X_train,y_train)  #Train the model 

pred = model.predict(X_test) # Predict the output for testing the model

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,pred) #Check accuracy score of the model
print('\nModel accuracy: ',accuracy)  

print('\n')
#--------------User's side----------------------

# Enter the values which you want to check
SepalLength=float(input('Enter sepal length in cm '))
SepalWidth=float(input('Enter sepal width in cm '))
PetalLength=float(input('Enter Petal length in cm '))
PetalWidth=float(input('Enter Petal width in cm '))

#Convert the data into a dataframe
data = {
    'SepalLengthCm':SepalLength,
    'SepalWidthCm':SepalWidth,
    'PetalLengthCm':PetalLength,
    'PetalWidthCm':PetalWidth,
}

test_df = pd.DataFrame([data])

# Predict the output given by model
predicted_result=model.predict(test_df)

# Print the output
result = ''.join(predicted_result)
print('\nPredicted result: ',result)


