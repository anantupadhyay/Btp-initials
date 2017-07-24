from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import math
import numpy as np
import pandas as pd 
import csv
import re
import sys

def tokenize(s):
	tokens = re.split('; |, |\-|\/|[_.http:www123456789]',s)
	while '' in tokens:
		tokens.remove('')
	tok = [s for s in tokens if len(s) >= 3]
	return tok

df = pd.read_csv('test.csv', header=0)
print (df.shape)

clean_train_reviews = []
num_site = df['URLofSite'].size
print (num_site)

for index in range(0, num_site):
	tokens = tokenize(df['URLofSite'][index])
	clean_train_reviews.append(tokens)


print ("Creating the bag of words.\n")
vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

test = pd.read_csv('test1.csv', header=0)
print (test.shape)

num_site = len(test['URLofSite'])
test_cat = []

print ("\nStarting to make some predictions\n")
for i in range(0, num_site):
	num_site_tok = tokenize(test['URLofSite'][i])
	test_cat.append(num_site_tok)

test_data_features = vectorizer.transform(test_cat)
test_data_features = test_data_features.toarray()

clf1 = RandomForestClassifier(n_estimators=30)
clf1 = clf1.fit(train_data_features, df["Category"])
result = clf1.predict(test_data_features)
acc = accuracy_score(test['Category'], result)
print (acc)


clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(train_data_features, df['Category'])
print ("Data is fit finally\n")
pre2 = clf.predict(test_data_features)
acc3 = accuracy_score(test['Category'], pre2)
print (acc3)

clf2 = LogisticRegression()
clf2.fit(train_data_features, df['Category'])
pre = clf2.predict(test_data_features)
acc2 = accuracy_score(test['Category'], pre)
print (acc2)

#	 		CREATING THE ENSEMBLE 			#

print ("Building the Ensemble")
eclf = VotingClassifier(estimators=[
	('rf', clf1), ('knn', clf), ('lr', clf2)], voting='soft')

eclf = eclf.fit(train_data_features, df['Category'])

print ("Predicting Using Ensemble")
fres = eclf.predict(test_data_features)
facc = accuracy_score(test['Category'], fres)
print (facc)