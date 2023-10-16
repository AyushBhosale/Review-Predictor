import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
nlp=spacy.load('en_core_web_lg')
# def remove_stopwords(text):
#     doc=nlp(text)
#     words=[]
#     for word in doc:
#         if word.is_stop:
#             continue
#         words.append(word)
#     listToStr = ' '.join(map(str, words))
#     return listToStr
dataset=pd.read_csv('reviews_data.csv')
dataset.info()
dataset.dropna(inplace=True)
print(dataset.isnull().sum())
dataset=dataset.drop(['location','Date','Image_Links','name'],axis=1)
print(dataset.head())
dataset['vector']=dataset['Review'].apply(lambda x:nlp(x).vector)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset['vector'],dataset['Review'],test_size=0.1,random_state=2022)
X_train_2d=np.stack(X_train)
X_test_2d=np.stack(X_test)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X_train_2di=sc.fit_transform(X_train_2d)
X_test_2di=sc.fit_transform(X_test_2d)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_2di,y_train)
y_pred=model.predict(X_test_2di)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
