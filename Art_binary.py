from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import math
import numpy as np
import pandas as pd 
import csv
import re
import sys
from sklearn import svm
from sklearn.ensemble import (
    BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
)
vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, max_features = 1000)

def word2ngrams(s, n=5):
	text = re.sub('[^a-zA-Z]+', '', s)
	return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]

def train(cat,clf):
	df = pd.read_csv('train_'+cat+'.csv', header=0)
	#print (df.shape)

	clean_train_reviews = []
	num_site = df["URLofSite"].size
	#print (num_site)

	for index in range(0, num_site):
		tokens = word2ngrams(df['URLofSite'][index])
		clean_train_reviews.append(tokens)


	#print ("Creating the bag of words.\n")
	global vectorizer

	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	train_data_features = train_data_features.toarray()

	#clf = MultinomialNB()
	clf.fit(train_data_features, df['Category'])
	return clf

def test(clf,cat):
	test = pd.read_csv('test14.csv', header=0)
	#print (test.shape)
	global vectorizer
	num_site = len(test["URLofSite"])
	test_cat = []

	#print ("\nStarting to make some predictions\n")
	for i in range(0, num_site):
		num_site_tok = word2ngrams(test["URLofSite"][i])
		test_cat.append(num_site_tok)

	test_data_features = vectorizer.transform(test_cat)
	test_data_features = test_data_features.toarray()
	result = clf.predict(test_data_features)
	#print (result)
	sahi = 0
	galat = 0
	for x in range(len(result)):
		if result[x] == cat:
			if test['Category'][x] == cat:
				sahi += 1
			else:
				galat += 1
		else:
			if test['Category'][x] == cat:
				galat += 1
			else:
				sahi += 1
	global su
	su += (((sahi*1.0)/(sahi+galat))*100)
	print (((sahi*1.0)/(sahi+galat))*100)
	print ("\n")
'''	fi = open('test14.csv','w', newline='')
	writer = csv.writer(fi, quoting=csv.QUOTE_ALL)
	writer.writerow(['URLofSite','Category'])
	fi.close()
	fi = open('test14.csv','a', newline='')
	writer = csv.writer(fi)

	for x in range(len(result)):
		if result[x] != cat or test['Category'][x] != cat:
			writer.writerow([test['URLofSite'][x],test['Category'][x]])
	fi.close()
	'''
fi = open('test14.csv','w', newline='')
writer = csv.writer(fi,quoting=csv.QUOTE_ALL)
writer.writerow(['URLofSite','Category'])
fi.close()
f =  open('test14.csv','a', newline='')
su = 0
writer = csv.writer(f)
r = open('test.csv','r')
reader = csv.reader(r)
for row in reader:
	writer.writerow(row)
f.close()
r.close()
cat_arr = ['Shopping','Business','Computers','Adult','Health','Society']
clf1 = MultinomialNB()
clf2 = RandomForestClassifier(n_estimators=30)
clf3 = KNeighborsClassifier()
clf4 = LogisticRegression()
clf5 = svm.SVC()
classifiers = [clf1,clf2,clf4,clf3,clf5]
#clf1 = MultinomialNB()

mxsu = 0
clf_array = []
for ty in cat_arr:
	su = 0
	for cs in classifiers:
		print ("The classifier is ", cs)
		print ('IN Catogory',ty)
		clf = (train(ty, cs))
		test(clf1,ty)
	print ("The average acc of ",cs, "is ", su/len(cat_arr))
	mxsu += su/len(cat_arr)

print ("The overall acc is ", mxsu)


acc = accuracy_score(test['Category'], result)
print (acc)




#	 		CREATING THE ENSEMBLE 			#
'''
print ("Building the Ensemble")
eclf = VotingClassifier(estimators=[
	('rf', clf1), ('knn', clf), ('lr', clf2)], voting='soft')

eclf = eclf.fit(train_data_features, df['Category'])

print ("Predicting Using Ensemble")
fres = eclf.predict(test_data_features)
facc = accuracy_score(test['Category'], fres)
print (facc)
'''