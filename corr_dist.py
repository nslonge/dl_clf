import os
import numpy as np

# defines a correlation distance classifier
class CorrDist:
	def __init__(self):
		self.avg_vectors = None
		self.classes = None
		self.d_vectors = None
		self.d_classes = None
		self.trained = False

	def fit(self, d_vectors, d_classes):
		self.d_vectors = d_vectors
		self.d_classes = d_classes
		self._set_classes()
		self._set_avg_tables()
		self.trained = True

	def _set_classes(self):
		cls = sorted(self.d_classes)
		self.classes = [[cls[0],1.0]]
		for cl in cls[1:]:
			if self.classes[-1][0] == cl: self.classes[-1][1]+=1.0
			else: self.classes.append([cl, 1.0])

	def _set_avg_tables(self):
		n = len(self.classes)
		m = self.d_vectors.shape[1]
		self.avg_vectors = np.zeros(shape=(n,m))
		
		# sum all data by class
		for cl, vec in zip(self.d_classes, self.d_vectors):
			self.avg_vectors[cl] += vec	
			
		# normalize
		self.avg_vectors[cl] /= self.classes[cl][1]

	def _get_corr_dist(self, v1, v2):
		n = v1.shape[0]
		avg1 = np.mean(v1)
		sdev1 = np.std(v1)
		avg2 = np.mean(v2)
		sdev2 = np.std(v2)

		v1_avg = np.full((n,),avg1)
		v1_sdev = np.full((n,),sdev1)
		v2_avg = np.full((n,),avg2)
		v2_sdev = np.full((n,),sdev2)

		return np.dot(((v1-v1_avg)/v1_sdev), ((v2-v2_avg)/v2_sdev))

	def predict(self, t_vectors):
		preds = np.zeros(shape=(t_vectors.shape[0],))
		for i, t_vec in enumerate(t_vectors):
			pred = []
			for cl, avg in enumerate(self.avg_vectors):
				pred.append((cl, self._get_corr_dist(t_vec, avg)))
			pred = sorted(pred, key=lambda x: x[1])
			preds[i] = pred[-1][0]
		return preds

	def score(self, t_vectors, t_classes):
		preds = self.pred(t_vectors)
		tot, correct = 0.0,0
		for pred, cl in zip(preds, t_classes):
			tot+=1.0
			if pred == cl: correct+=1
		return correct/tot
