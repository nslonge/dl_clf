from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import subprocess
import math

# load list of vowels
fp = open('vowels.txt', 'rb')
vowels = fp.read().strip().split('\n')
fp.close()

############################ METHODS FOR FORMANTS #########################
def read_formant(infile):
# method borrow from Christian Herbst
	cnt = 0
	maxnFormants = None
	insideDataStructure = False
	dataCnt = 0
	intensity = None
	nFormants = None
	frequency = None
	bandwidth = None
	frameIdx = 0
	formant_slice = []
	formant_arr = []
	time_arr = []
	nx,dx,x1,maxnFormants = 0,0,0,0

	f = open(infile, 'rb')
	
	for ln in f:			
		cnt += 1
		txt = ln.strip()
		if cnt == 6: # number of frames
			nx = int(txt)
		elif cnt == 7: # dx
			dx = float(txt)
		elif cnt == 8: # x1
			x1 = float(txt)
		elif cnt == 9: # maxnFormants
			maxnFormants = float(txt)
		elif cnt > 9:
			if insideDataStructure:
				dataCnt += 1
				if dataCnt == 1:
					nFormants = int(txt)
				else:
					tmpCnt = dataCnt - 2
					formantCount = tmpCnt / 2 + 1
					if tmpCnt % 2 == 0:
						frequency = float(txt)
					else:
						formant_slice.append(frequency)
						if formantCount == nFormants:
							# add the data here
							x = x1 + dx * (frameIdx - 1)
							time_arr.append(x)
							formant_arr.append(formant_slice)
							insideDataStructure = False
			else:
				dataCnt = 0
				insideDataStructure = True
				formant_slice = []
				intensity = float(txt)
				frameIdx += 1
				
	f.close()
	
	# check the data
	if not len(time_arr) == len(formant_arr):
		raise Exception("data array sizes don't match!")
	if not nx == len(time_arr):
		raise Exception('file "' + infile + '" promised to contain ' + \
			str(nx) + ' frames, but only ' + str(len(time_arr)) + \
			' were found')
	# return data
	return time_arr, formant_arr

def call_praat(infile,f_max=5000,window_length=0.025,time_step = 0):
# method borrowed from Christain Herbst
	max_formants = 5 
	pre_emph = 50 # [Hz]

	name = '.'.join(infile.split('/')[-1].split('.')[:-1])
	path = '/'.join(infile.split('/')[:-1]) + '/'
	
	# create Praat script
	control_file = path + 'tmp.praat'
	f = open(control_file, 'w')
	f.write('Read from file... ' + infile + '\n')
	cmd = 'To Formant (burg)... ' + str(time_step) + ' ' + str(max_formants) \
		+ ' ' + str(f_max) + ' ' + str(window_length) + ' ' + str(pre_emph)
	f.write(cmd + '\n')
	formantDataFileName = path + name + '.Formant'
	f.write('Write to short text file... ' + formantDataFileName + '\n')
	f.close()

	# call Praat on script
	args = ['Praat', control_file]
	#msg = subprocess.Popen(args, stdout=subprocess.PIPE, 
	#					   stderr=subprocess.PIPE).communicate()	
	msg = subprocess.call(args)	

	# delete controlfile
	os.remove(control_file)

	#return readFile(formantDataFileName)

def get_formants(filename, v_info, sex):
	# input: abs path to file, v_info = (phone, str_t, end_t), sex = spkr sex
	# output: (lst1,lst2), where lst1 = time slices, lst2 = formants
	
	vowel = v_info[0]
	str_t = v_info[1]
	end_t = v_info[2]
	ln = end_t-str_t

	# get name identifier	
	name = filename.split('.')[0]
	path = '/'.join(filename.split('/')[:-1])

	# check if there is already a formant file present
	form_lst = []

	# have to create a unique name for each formant file
	frm_path = name + vowel + str(str_t).replace('.','')[:4] 

	try: 
		fp = open(os.path.join(path, frm_path +'.Formant'), 'rb')
		fp.close()
	# no formant data, so calculate it
	except : 
		# read wavfile
		(rate, sig) = wav.read(filename)

		# isolate vowel, +-.025 because praat sampling starts/end at +-.025s
		str_frame = int(rate * max(str_t-.025,0))
		end_frame = int(rate * (end_t+.025))

		# write temporary vowel file (because Praat sucks ass)
		tmp_wav = os.path.join(path,frm_path + '.wav')
		wav.write(tmp_wav,rate,sig[str_frame:end_frame+1])	
	
		fm = 5000
		if sex == 0:
			fm = 5500

		# call praat
		call_praat(tmp_wav,f_max=fm,window_length=.025,time_step=.01) 

		# delete temporary vowel file
		os.remove(tmp_wav)	

	# read formants
	times, formants = read_formant(os.path.join(path,frm_path+'.Formant'))

	# compute averages for 1st/2nd half of vowel
	form_arr=[np.zeros(shape=(3,)),np.zeros(shape=(3,))]
	cnt = [0,0]

	# ignore first and last samples
	for t, fs in zip(times[1:-1], formants[1:-1]):
		t = t-.025
		i = 1-int(t < ln/2.0)
		form_arr[i] += np.array(fs[:3])
		cnt[i] += 1

	# take averages	
	f = np.vectorize(lambda x: x/cnt[0])
	g = np.vectorize(lambda x: x/cnt[1])
	form_arr[0] = f(form_arr[0]) 
	form_arr[1] = g(form_arr[1])
	return np.concatenate((form_arr[0],form_arr[1]),axis=0)

def normalize(spkr_fs):
	# apply formant normalization, following Adank
	# Fi_new = (Fi_old - Fi_avg)/Fi_sdev, with Fi_avg and Fi_sdev computed over
	#								  all Fi ocurrences for that speaker
	
	# f1_i = [1,4]
	# f2_i = [2,5]
	# f3_i = [3,6]
	
	# get averages
	f_avg = np.zeros(shape=(3,))
	cnt = 0
	for word,segs in spkr_fs.iteritems():
		for seg in segs:
			f_avg += seg[1:4]
			f_avg += seg[4:7]
			cnt+=2
	f = np.vectorize(lambda x:x/float(cnt))
	f_avg = f(f_avg)		

	# get sdev
	f_sdev = np.zeros(shape=(3,))
	cnt = 0
	pow2 = np.vectorize(lambda x: pow(x,2))
	for word,segs in spkr_fs.iteritems():
		for seg in segs:
			tmp = np.zeros(shape=(3,))
			tmp = pow2(seg[1:4] - f_avg)
			f_sdev+=tmp
			tmp = np.zeros(shape=(3,))
			tmp = pow2(seg[4:7] - f_avg)
			f_sdev+=tmp
			cnt+=2
	f = np.vectorize(lambda x: x/float(cnt))
	root = np.vectorize(lambda x: math.sqrt(x))
	f_sdev = root(f(f_sdev))

	# apply normalization
	for word, segs in spkr_fs.iteritems():
		for seg in segs:
			seg[1:4] = (seg[1:4] - f_avg)/(f_sdev)
			seg[4:7] = (seg[4:7] - f_avg)/(f_sdev)

##############################################################################

def get_mfcc(filename, v_info):
	# input: abs_path to file, v_info = (phone, str_t, end_t)
	# output: 1x26 array with [:12] 1st half avg mfcc, [13:] 2nd half avg mfcc

	vowel = v_info[0]
	str_t = v_info[1]
	end_t = v_info[2]

	# read wavfile
	(rate, sig) = wav.read(filename)

	# isolate vowel
	str_frame = int(rate * (str_t-.025))
	end_frame = int(rate * (end_t+.025))

	ln = (end_t-str_t) 

	# cut down wav to vowel
	vowel_sig = sig[str_frame:end_frame+1]
	wav.write('test.wav', rate, vowel_sig)
	
	# compute MFCCs
	mfccs = mfcc(vowel_sig, rate, winstep=0.005)

	# array for storing mfcc avgs
	size = len(mfccs[0])
	mfcc_arr=[np.zeros(shape=(size,)),np.zeros(shape=(size,))]

	# sum MFCCs for first and second half of vowel
	cnt = [0.0,0.0]
	for i, features in enumerate(mfccs[4:-4]):
		elapsed = .005*i+.01  # indices mark .01s ints
		half = 1 - int(elapsed < ln/2.0) # first or second half of vowel
		mfcc_arr[half] += features
		cnt[half]+=1
	
	# take averages
	mfcc_arr[0] /= cnt[0]
	mfcc_arr[1] /= cnt[1] 

	# append avg lists
	return np.concatenate((mfcc_arr[0],mfcc_arr[1]),axis=0)

def load_segment_file(filename):
	# input: filename of file containing utterance segmentation data
	# output: list of segment information: 
	#		  word_lst = [(word,[(vowel, str_t, end_t), ...])]

	word_lst = [] 	# [(word,[(vowel, str_t, end_t), ...])]

	in_word = False
	first = True
	cur_word = ''
	cur_lst = []

	fp = open(filename, 'rb')
	for ln in fp:
		ln = ln.strip().replace('\n','').split(' ')
		# sp marks start of word
		if ln[0] == 'sp':
			# set what we've counted
			if not cur_word == '': 
				word_lst.append((cur_word,cur_lst))
			cur_word = ''
			cur_lst = []
			in_word = False	
			continue
		# make sure we're dealing with a word
		if len(ln) == 1:
			in_word = True
			cur_word = ln[0]	
			continue
		if in_word:
			if not ln[0] in vowels: continue
			cur_lst.append(ln)		

	fp.close()	
	return word_lst

def get_distance(v1, v2):
	return math.sqrt(np.dot((v1-v2),(v1-v2)))



