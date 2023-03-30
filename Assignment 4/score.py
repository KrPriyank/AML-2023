#imports
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

#data and splitting 
rawdata = pd.read_csv("rawdata.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
validate = pd.read_csv("validate.csv")

X_train= train.Text
Y_train= train.Label
X_validate = validate.Text
Y_validate= validate.Label
X_test = test.Text
Y_test = test.Label

cv = CountVectorizer().fit(rawdata.Text)

X_train = cv.transform(X_train)
X_val = cv.transform(X_validate)
X_test = cv.transform(X_test)

tfidf = TfidfTransformer()

tfidf_train = tfidf.fit_transform(X_train)
tfidf_val = tfidf.fit_transform(X_val)
tfidf_test = tfidf.fit_transform(X_test)

Y_train = Y_train.astype('int')
Y_validate = Y_validate.astype('int')
Y_test = Y_test.astype('int')

def text_to_vec(text):
    obs = cv .transform([text])
    obs = tfidf.fit_transform(obs)
    return obs

fname = open("mlp",'rb')
mlp = pickle.load(fname)

def score(text:str, model, threshold:float=0.5) -> (bool,float):
    # Transforming the input text
    tv = text_to_vec(text)
    print(tv.shape)

    # Predicting the propensity score
    prediction = model.predict(tv)
    propensity = model.predict_proba(tv)
    return prediction[0], propensity[0]

print(score("You have won a free voucher of $10000, click on the link to redeem",mlp,0.5))