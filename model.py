
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle

data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# print(data_fake.head())
# print(data_true.head())


data_fake["class"] = 0
data_true["class"] = 1

print(data_fake.shape)
print(data_true.shape)

## Manual Testing For Both The Dataset

data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis=0, inplace=True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace=True)


# Assigning classes to the dataset
data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1

data_merge = pd.concat([data_fake, data_true], axis=0)
# print(data_merge.head())

# Droping Unwanted Columns 
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# print(data.head(9))
# print(data.shape)

# function to clear the text data
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://S+|www.\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]'%re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

# Training dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(xv_train, y_train)

pred_model = model.predict(xv_test)
model.score(xv_test, y_test)
print(classification_report(y_test, pred_model))

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    user_sample_test = pd.DataFrame(testing_news)
    user_sample_test['text'] = user_sample_test['text'].apply(wordopt)
    new_x_test = user_sample_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred = model.predict(new_xv_test)

    return print("\n Prediction : ", format(output_label(pred[0])))

# sample = str()

# user_input = input("Enter a text to test : ")
# sample = str(user_input)

# manual_testing(sample)

# import pickle

# Save the model to a pickle file
with open('model1.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorization to a pickle file
with open('vector.pkl', 'wb') as vector_file:
    pickle.dump(vectorization, vector_file)
