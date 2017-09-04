from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, max_features=1000)


def word2ngrams(s, n):
    text = re.sub('[^a-zA-Z]+', '', s)
    return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]


def train(cat, clf):
    df = pd.read_csv('train_' + cat + '.csv', header=0)
    # print (df.shape)

    clean_train_reviews = []
    num_site = df["URLofSite"].size
    # print (num_site)

    for index in range(0, num_site):
        tokens = word2ngrams(df['URLofSite'][index], 6)
        #tokens.extend(word2ngrams(df['URLofSite'][index], 6))
        tokens.extend(word2ngrams(df['URLofSite'][index], 7))
        clean_train_reviews.append(tokens)
        

    # print ("Creating the bag of words.\n")
    global vectorizer

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    # clf = MultinomialNB()
    clf.fit(train_data_features, df['Category'])

    test = pd.read_csv('test.csv', header=0)
    # print (test.shape)

    num_site = len(test["URLofSite"])
    test_cat = []

    # print ("\nStarting to make some predictions\n")
    for i in range(0, num_site):
        num_site_tok = word2ngrams(test["URLofSite"][i],6)
        
        #num_site_tok.extend( word2ngrams(test["URLofSite"][i],6))
        num_site_tok.extend( word2ngrams(test["URLofSite"][i],7))
        test_cat.append(num_site_tok)

    test_data_features = vectorizer.transform(test_cat)
    test_data_features = test_data_features.toarray()
    result = clf.predict(test_data_features)
    # print (result)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for x in range(len(result)):
        if result[x] == cat:
            if test['Category'][x] == cat:
                tp += 1
            else:
                fp += 1
        else:
            if test['Category'][x] == cat:
                fn += 1
            else:
                tn += 1
    global su
    su = 0
    #  print
    #  tp, fp, tn, fn
    acc = (((tp + tn) * 1.0) / (tp + tn + fp + fn)) * 100
    print(    "The acc is ", acc, "\n")
    su += acc
    return clf
    #print("\n")

fi = open('test14.csv','w', newline='')
writer = csv.writer(fi, quoting=csv.QUOTE_ALL)
writer.writerow(['URLofSite', 'Category'])
fi.close()
f =  open('test14.csv','a', newline='')
su = 0
writer = csv.writer(f)
r = open('test.csv', 'r')
reader = csv.reader(r)
for row in reader:
    writer.writerow(row)
f.close()
r.close()
cat_arr = ['Shopping', 'Business', 'Computers', 'Adult', 'Health', 'Society']
estimator = 10
dt = LogisticRegression()
clf1 = RandomForestClassifier(n_estimators=estimator)
clf2 = GradientBoostingClassifier(n_estimators=estimator, learning_rate=1.0, max_depth=2)
clf3 = AdaBoostClassifier(n_estimators=estimator, base_estimator=dt, learning_rate=1)

#clf1 = MultinomialNB()


a1, a2, a3, a4 = 0, 0, 0, 0
for cat in cat_arr:
    print('In Category ', cat)
    #print ('time_pass')
    print("RandomForest " )
    c1 = train(cat, clf1)
    a1 += su
    print("LDABoosting ",end = '')
    c2 = train(cat, clf2)
    a2 += su
    print("AdaBoost ", end = ''  )
    c3 = train(cat, clf3)
    a3 += su
    print ('MultinomialNB()',end = '')
    c4 = train(cat,MultinomialNB())
    print("Building Voting Classifier")
    eclf = VotingClassifier(estimators=[('lda', c1), ('rf', c2), ('ada', c3)], voting='soft')
    ec = train(cat, eclf)
    a4 += su

print("The overall acc of ", clf1, " is", a1 / 6)
print("The overall acc of ", clf2, " is", a2 / 6)
print("The overall acc of ", clf3, " is", a3 / 6)
print("The overall acc of Voting Classifier", " is", a4 / 6)