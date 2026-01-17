import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
<<<<<<< HEAD:Naive/naive_bayes.py
=======
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.plot import plot_confusion_matrix

data = pd.read_csv('D:\\kuliah\\MCL\\Naive\\data.csv');

data_awal = data.drop(columns="Play")
data_label = data.Play
data_awal_train, data_awal_test,data_label_train, data_label_test = train_test_split(
    data_awal, 
    data_label,
    test_size=0.2,
    random_state=42)

print("Spliting: ",[
    data_awal_train.shape,
    data_awal_test.shape,
    data_label_train.shape,
    data_label_test.shape
])

preprocessor = ColumnTransformer([
    ('categoric', cat_pipe(encoder="onehot"),["Outlook","Temperature","Humidity","Windy"])
])

# print(preprocessor)
from sklearn import pipeline
>>>>>>> fa93b24b3d9d58d25836774950edf4a5e51b9b37:MCL/Naive/naive_bayes.py
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_excel('credit.xlsx')
x = data.iloc[:, 0:4]
y = data.iloc[:, 4]

# print("Features:\n", x)
# print("Info \n", data.info())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=123    
)
# print("Spliting: \n",[
#     x_train,
#     x_test
# ])

model = GaussianNB()
nb_model = model.fit(x_train, y_train)
y_pred = nb_model.predict(x_test)

print("Class Count: \n", nb_model.class_count_)
print("Hasil Prediksi: \n", y_pred)