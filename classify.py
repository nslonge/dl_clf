""" Drive file for dialect classification. Usage:

	1. Run the classification system on the test data, using held-out speaker
	   to test accuracies. 
	
		python classify.py -t
	
	2. Speaker input mode: asks the speaker to record themselves, then 
	   classifiers their dialect.

		python classify.py 

"""

import numpy as np
import os
import sys
import string
import math
import random

### Scikit-learn classifiers ###
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
###################################################

### local files ###
from corr_dist import CorrDist
from speaker import Speaker
from new_speaker import process_new_speaker
import utils
###################

# load list of words to use in classification
fp = open('words.txt', 'rb')
words = fp.read().strip().split('\n')
fp.close()

dl_names = ['at','ne','no','so','we']

#inds = [0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,26,29,31,33,34,35]

def test():
	# filpaths
	cur_path = os.getcwd()
	spkr_path = os.path.join(cur_path, 'train_data')

	# load speaker information	
	spkrs = []
	for spkr in os.listdir(spkr_path):
		if not len(spkr) == 3: continue
		if spkr[:-1] == 'mi': continue

		# initialize speaker, load segments
		dl = dl_names.index(spkr[:-1])
		sex = int(spkr[-1]) in [0,6,7,8,9]
		spkr = Speaker(os.path.join(spkr_path, spkr), spkr, sex, dl) 	
		spkr.load_features('cvc')
		#spkr.load_features('vowel')
		#spkr.load_features('hpspin')
		spkr.load_distance_table(words)	
		spkrs.append(spkr)

	# initialize classifiers
	classifiers = [
		CorrDist(),	# ~75% 
        SVC(kernel="linear", C=1), # ~65%
		LogisticRegression(C=10),
        SVC(kernel="poly", degree=2, C=1), #~70%
        LinearSVC()] # ~80% 

	names = ["Correlation Distance",
             "Linear SVM",
			 "Logistic Regression",
			 "Poly SVM",
			 "Linear SVC"]

	score = []
	spkr_pred = {name: [] for name in names}
	# since we have so few speakers, use one vs. rest to test
	for name, clf in zip(names, classifiers):
		correct, tot = 0,0
		for spkr in spkrs:
			# setup test data for other speakers
			tst_info = spkr.d_vector.reshape(1,spkr.d_vector.shape[0])
			tst_class = np.array([spkr.dialect]) 

			trn_info = np.array([s.d_vector for s in spkrs if s <> spkr])
			trn_class = np.array([s.dialect for s in spkrs if s <> spkr])

			if name == 'Linear SVM' or name == 'Linear SVC':
				scaler = preprocessing.StandardScaler().fit(trn_info)
				trn_info = scaler.transform(trn_info)
				tst_info = scaler.transform(tst_info)

			clf.fit(trn_info, trn_class)
			pred = clf.predict(tst_info)
			#spkr_pred[spkr].append(np.array([int(dl==pred) for dl in dialects]))
			#print pred, spkr.dialect
			
			if pred == spkr.dialect:
				correct+=1
			else:
				spkr_pred[name].append(spkr.name)
			tot+=1
		#print name
		#print float(correct)/float(tot)
		score.append([name,float(correct)/float(tot)])

	print "Percent speakers correctly classified, by classifier:\n"	
	for name,sc in score:
		print name, sc

def load_classifiers():
	cur_path = os.getcwd()
	spkr_path = os.path.join(cur_path, 'train_data')

	# load speaker information	
	spkrs = []
	for spkr in os.listdir(spkr_path):
		if not len(spkr) == 3: continue
		if spkr[:-1] == 'mi': continue

		# initialize speaker, load segments
		dl = dl_names.index(spkr[:-1])
		sex = int(spkr[-1]) in [0,6,7,8,9]
		spkr = Speaker(os.path.join(spkr_path, spkr), spkr, sex, dl) 	
		spkr.load_features('cvc')
		#spkr.load_features('vowel')
		#spkr.load_features('hpspin')
		spkr.load_distance_table(words)	
		spkrs.append(spkr)

	# initialize classifiers
	classifiers = [
		CorrDist(),	# ~75% 
        #KNeighborsClassifier(5), #~50% 
        SVC(kernel="linear", C=1), # ~65%
		LogisticRegression(C=10),
        SVC(kernel="poly", degree=2, C=1), #~70%
        LinearSVC()] # ~80% 
        #DecisionTreeClassifier()] #~65%

	names = ["Correlation Distance",
			 #"Nearest Neighbors",
             "Linear SVM",
			 "Logistic Regression",
			 "Poly SVM",
			 "Linear SVC"]
             #"Decision Tree"] 

	clf_lst = []
	
	trn_info = np.array([s.d_vector for s in spkrs])
	trn_class = np.array([s.dialect for s in spkrs])

	for name, clf in zip(names, classifiers):
		# setup test data for other speakers
		scaler = None
		if name == 'Linear SVM' or name == 'Linear SVC':
			scaler = preprocessing.StandardScaler().fit(trn_info)
			trn_info = scaler.transform(trn_info)

		clf.fit(trn_info, trn_class)
		clf_lst.append((name,clf,scaler))
	
	return clf_lst

def classify_speaker():
	dialects = ['Middle Atlantic','New England','Northern','Southern','Western']

	os.system('cls' if os.name == 'nt' else 'clear')
	name = raw_input("What's your first name?\n").lower()

	# get speaker data 
	spkr = process_new_speaker(name)
	d_vec = np.array([spkr.d_vector])

	# construct classifiers
	classifiers = load_classifiers()

	# classify	
	preds = []
	for name, clf, scaler in classifiers:
		print name
		if name == 'Linear SVM' or name == 'Linear SVC':
			preds.append(clf.predict(scaler.transform(d_vec))[0])
		else:
			preds.append(clf.predict(d_vec)[0])
		print int(preds[-1])
	
	# find best guess, with scaled weights
	weights = [.2,.1,.2,.1,.4] # how much to trust each classifer
	guess = [0.0,0.0,0.0,0.0,0.0]
	for i,pred in enumerate(preds):
		guess[int(pred)] += weights[i]	
	m1 = max(range(0,5), key = lambda x:guess[x])
	guess.pop(m1) 
	m2 = max(range(0,4), key = lambda x:guess[x])

	# print guesses
	print "Closest dialect: " + dialects[m1]
	if guess[m2] <> 0.0:
		print "2nd closest dialect: " + dialects[m2]


if __name__ == "__main__":
	if len(sys.argv) == 2 and sys.argv[-1] == '-t':
		test()
	elif len(sys.argv) == 1:
		classify_speaker()
	else:
		print "Improper use. See documentation"



