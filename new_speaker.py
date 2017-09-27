import os
import random
import time
import math
from multiprocessing import Process, Lock
from pydub import AudioSegment
from record import Recorder
from mod_align import align_external
from speaker import Speaker

# create text file for forced alignment
def create_text_files(path, name):
	# input: name of speaker, used to create assocaited text files
	# output: name1.txt, name2.txt, each with half of the 26 words to be rec'd

	fp = open('words.txt','rb')
	words = fp.read().strip().split('\n')
	random.shuffle(words)

	ws1 = words[:len(words)/2]
	ws2 = words[len(words)/2:]

	filename1 = os.path.join(path,name + '1.txt')
	filename2 = os.path.join(path,name + '2.txt')

	fp = open(filename1, 'wb')
	fp.write(' '.join(ws1))
	fp.close()

	fp = open(filename2, 'wb')
	fp.write(' '.join(ws2))
	fp.close()

	return ws1, ws2

def rec_prompt(filename, ws):
	# helper function for recording, to allow iterated rec's
	os.system('cls' if os.name == 'nt' else 'clear')
	print "Here the words to record. Please read them in order:"
	print
	print ' '.join(ws) 
	print
	start = raw_input("Press <Enter> to begin recording. Recording will stop after 1s of silence")

	while start <> '':
		print "Invalid command"
		start = raw_input("Press <Enter> to begin recording. Recording will stop after 1s of silence")

	# wait for .2s
	time.sleep(.3)
		
	rec = Recorder(filename)		
	rec.record_speech()
	while True:
		print  
		print "Would you like to record again?"
		y = raw_input("Type <y> + <Enter> for 'yes', <n> + <Enter> for 'no'\n")
		if y == 'y': 
			rec_prompt(filename,ws) 
			break
		elif y == 'n': 
			break 

def record_speaker(path, name, ws1, ws2):
	# prompts speaker to record requisite words, in two groups
	# Input: name = speaker id, ws1 = first half of words, ws2 = second half
	# Output: name1.wav = rec of ws1, name2.wav = rec of ws2

	os.system('cls' if os.name == 'nt' else 'clear')
	print "To classify your accent, I need you to read some words for me"
	print
	print "You'll be asked to read two sets of 13 words, in order"
	print
	cont = raw_input("Press <Enter> now to continue to recording")
	while cont <> '':
		print "Dont' type! Just press enter."
		cont = raw_input("Press <Enter> now to continue to recording")

	os.system('cls' if os.name == 'nt' else 'clear')
	print "Here's how recording works:"
	print "\t 1. I display the words to be read"
	print "\t 2. You press <Enter> to begin recording"
	print "\t 3. You read the words"
	print "\t 4. You press <Enter> or remain silent for 2s to halt recording"
	print "\t 5. I ask if you'd like to try again"
	print 
	cont = raw_input("Press <Enter> now to go to recording mode\n")
	while cont <> '':
		print "Dont' type! Just press enter."
		cont = raw_input("Press <Enter> now to go to recording mode\n")
	os.system('cls' if os.name == 'nt' else 'clear')

	rec_prompt(os.path.join(path,name+'1'), ws1)
	rec_prompt(os.path.join(path,name+'2'), ws2)

def forced_align(path, name, it):
	# runs the Penn forced aligner on speaker's utterances
	# Input: path = path to wav & txt files for speaker, name = speaker ID
	#		 it = iteration, 1 or 2
	# Output: name<it>_seg.txt = file with segment information for name<it>.wav 
	align_external((os.path.join(path,name + it + '.wav'),
					os.path.join(path,name + it + '.txt'),
					os.path.join(path,name + it + '.TextGrid')))

def process_new_speaker(name):

	path = os.path.join(os.path.join(os.getcwd(), 'spkr_data'), name)
	try: os.mkdir(path)
	except : pass

	# get txt and wav files for forced alignment
	ws1, ws2 = create_text_files(path,name)	
	record_speaker(path,name,ws1,ws2)

	# get forced alignment
	forced_align(path, name, '1')	
	forced_align(path, name, '2')	

	# initialize new speaker object	
	spkr = Speaker(path, name, 1, -1) 	
	spkr.load_features('')
	spkr.load_distance_table()	

	return spkr
