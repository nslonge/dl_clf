import os
import utils
import numpy as np
import string

MFCC_COUNT = 26

class Speaker:
	def __init__(self, path, name, sex, dialect):
		self.words = [] 		# list of words
		self._cnts = {} 		
		self.fs = {}			# word -> [V1, V2, ...], Vi = (MFCC_COUNT,)
		self.name = name
		self.path = path
		self.d_vector = None
		self.dialect = dialect
		self.sex = sex
	
	def load_distance_table(self,words=None):
		# inputs: words = [w1, ...], words to use in d_table
		# output: table 1 x |seg|*(|seg|-1)*(1/2) array w/ dist between segs
		

		if words == None:
			words = self.fs.iterkeys()
		words = sorted(words)
		words2 = list(words)
		words2.pop(0)

		# count number of unique segments
		n = sum([self.fs[word].shape[0] for word in words])

		# initialize table
		d_vector = np.zeros(shape=((n*(n-1))/2,))

		# populate tabel
		i = 0
		for word in words[:-1]:
			segs = self.fs[word]
			for seg in segs:
				for word2 in words2:
					segs2 = self.fs[word2]
					for seg2 in segs2:
						dist = utils.get_distance(seg,seg2)
						d_vector[i] = dist
						i+=1
			words2.pop(0)
		self.d_vector = d_vector
		if not self.d_vector.any():
			raise ValueError ("Speaker has no features loaded")

	def load_features(self, dir_name):
		path = os.path.join(self.path, dir_name)	# path to audio/txt data 
		files = os.listdir(path)

		fs = {}	

		process = True
		# first check if audio is pre-processed
		if self.name + '.dat' in files: 
			process = False
			self._load_file_data(os.path.join(path,self.name + '.dat'))
			return
	
		# otherwise, process audio
		fs = {}
		cnts = {}
		for fl in files:
			if not fl.endswith('_seg.txt'): continue

			# utterance ID
			name = fl.split('_')[0]					

			# ignore repeated utterances, those not in range
			if self._check_repeat(files, name): continue 

			# words = [(word,[(vowel, str_t, end_t)])]
			words = utils.load_segment_file(os.path.join(path, fl)) 

			for word,vowels in words:
				mfccs = np.zeros(shape=(MFCC_COUNT,))
					
				# only use monosyllabic words
				if len(vowels) > 1: continue 		
		
				# extract mel frequency cepstral co-effs 
				for vowel, str_t, end_t in vowels:
					if vowel == 'AH0' or vowel == 'ER0': continue 
					mfcc = utils.get_mfcc(os.path.join(path, name + '.wav'), 
								 (vowel, float(str_t), float(end_t)))
					if not mfccs.any():
						mfccs = mfcc.reshape(1,mfcc.shape[0])
					else:
						mfccs = np.vstack((mfccs, mfcc))

				# dont' add word if empty
				if not mfccs.any(): continue			

				# never seen this word
				if not word in fs:				
					fs[word] = mfccs
					cnts[word] = 1.0
				else:
					if not fs[word].shape == mfccs.shape:
						print "Vowel mismatch error"
						print name, word
						sys.exit()
					# add to average	
					fs[word] = (cnts[word] * fs[word] + mfccs)/(cnts[word]+1.0)
					cnts[word]+=1.0
		# save data
		self._save_dat(os.path.join(path,self.name+'.dat'), fs)
		# update global totals
		self._update_fs(fs,cnts)

	def _load_file_data(self, filename):
		fs = {}
		cnts = {}
		cur_word = ''
		cur_fs = []

		fp = open(filename, 'rb')
		for ln in fp:
			ln = ln.strip().replace('\n','')
			if ln.replace("'",'').isalpha():
				if cur_word <> '': 
					fs[cur_word] = np.array(cur_fs)
					cnts[cur_word] = 1.0 
				cur_word = ln
				cur_fs = []
				continue
			cur_fs.append(map(lambda x: float(x),ln.split(' ')))
		# add last word
		fs[cur_word] = np.array(cur_fs) 
		cnts[cur_word] = 1.0
		fp.close()
	
		self._update_fs(fs,cnts)

	def _update_fs(self,fs,cnts):
		# this is split off from the update in load_features to allow us to
		# save the data for each type of input individually
		for word,segs in fs.iteritems():
			if not word in self.fs:
				self.fs[word] = segs
				self._cnts[word] = cnts[word]
			else:
				self.fs[word] = (self._cnts[word] * self.fs[word] + 
								 		fs[word] * cnts[word])/\
								(self._cnts[word] + cnts[word])
				self._cnts[word] += cnts[word]
	
	def _save_dat(self,filename, fs):
		fp = open(filename, 'wb')
		words = sorted(fs.iterkeys())
		for word in words:
			fp.write(word + '\n')
			for seg in fs[word]:
				seg = seg.tolist()
				fp.write(reduce(lambda x,y: str(x)+' '+str(y),seg)+'\n')
		fp.close()	

	def _check_repeat(self, files, name):
		# check if name.wav was repeated
		# original: at1D0.wav, repeated: at1D1000.wav
		#			at1D10.wav, repeated: at1D1010.wav
		#			at1D1030.wav, repeated: at1D2030.wav, etc.

		# get segment id number
		int_id = name.translate(None,string.letters)[1:]
		if len(int_id) == 1: 
			int_id = '100'+int_id
		elif len(int_id) == 2:
			int_id = '10'+int_id
		elif len(int_id) == 3:
			int_id = '1'+int_id
		elif len(int_id) == 4: 	
			int_id = str(int(int_id[0])+1) + int_id[1:]

		# generate repeat name
		rep_name = name[:4]+int_id+'.wav'	

		# check if repeated
		if rep_name in files:
			return True
		return False

	def _in_range(self, inds,name):
		i = name.translate(None,string.letters)[1:] 	
		# found a repeated 1,2,3 digit input
		if len(i) == 4:
			i = i[1:]
			while i[0] == '0':
				if len(i) == 1: break 
				i = i[1:]
			if int(i) in inds:
				return True
		else:
			if int(i) in inds:
				return True
		return False
