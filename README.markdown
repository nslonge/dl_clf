#dialect_clf
Automatic dialect classification system for American English.

Classifiers speakers into one of five dialect regions:
	-see https://u.osu.edu/nspcorpus/

1. Middle Atlantic
2. New England
3. Inland North
4. Southern
5. Western

These regions are based on the dialect regions described by Labov, Ash, 
and Boberg (2006) in their `Atlas of North American English.' 

## Quickstart

1. Run the classification system on the test data, using held-out speaker   
   to test accuracies.                                                      
 
		`python classify.py -t`                                                  
2. Speaker input mode: prompt user to record themselves, then classifies 
   their dialect.                                               

		`python classify.py`   

##Dependencies
	- [numpy](http://www.numpy.org/)
	- [scipy](https://www.scipy.org/)
	- [scikit-learn](http://scikit-learn.org/stable/)
	- [pydub](https://github.com/jiaaro/pydub)
	- [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)
	- [python_speech_features](https://github.com/jameslyons/python_speech_features)

##System Design
The system utilizes the ACCDIST dialect identification metric 
(Huckvale 2007) to associate speakers with a unique vector representing
their particular dialect. Once these vectors are computed, the system builds
five classifiers:
1. Correlation distance (Huckvale 2007)
2. Linear SVM
3. Logisitic Regression
4. Polynomial SVM
5. Linear SVM w/ liblinear (http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)  

The classifiers are trained on the "The Nationwide Speech Project Corpus"
(Clopper & Pisoni 2006), which comprises five male and five female speakers
from each dialect region.

Using a held-out test procedure, the system acheives an accuracy of ~85%. 
