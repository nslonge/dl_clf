""" Class for recording to wav. This code is adapted from stackoverflow, but at
	I can't find the post I borrowed from. TODO: cite properly"""

import os
import pyaudio
import wave
import audioop
from collections import deque
import time
import math
import inspect

class Recorder():
	def __init__(self,filename):
		self.CHUNK = 1024  # CHUNKS of bytes to read each time from mic
		self.FORMAT = pyaudio.paInt16
		self.CHANNELS = 1
		self.RATE = 11025
			
		self.SILENCE_LIMIT = 2  
		self.PREV_AUDIO = 0.5  
		
		self.THRESHOLD = 300
		self.num_phrases = -1
		self.filename = filename

	def record_speech(self, path='./'): 
		#Open stream
		p = pyaudio.PyAudio()
		stream = p.open(format=self.FORMAT, 
						channels=self.CHANNELS, 
						rate=self.RATE, 
						input=True, 
						frames_per_buffer=self.CHUNK)

		audio2send = []
		cur_data = ''  # current chunk of audio data
		rel = self.RATE/self.CHUNK
		slid_win = deque(maxlen=self.SILENCE_LIMIT * rel)
		#Prepend audio from 0.5 seconds before noise was detected
		prev_audio = deque(maxlen=self.PREV_AUDIO * rel)
		started = False
		while True:
			cur_data = stream.read(self.CHUNK)
			slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))

			sm = sum([x > self.THRESHOLD for x in slid_win])
			if sm > 0:
				if started == False:
					started = True
					print "Started recording"
				audio2send.append(cur_data)
	
			elif started:
				print "Finished recording"
				filename=self.save_speech(list(prev_audio)+audio2send, 
										  self.filename, p)
                # Reset all
				break
			else:
				#print("Recording")
				prev_audio.append(cur_data)

		stream.close()
		p.terminate()
		return filename

	def save_speech(self, data, filename, p):
		"""
		Saves mic data to temporary WAV file. Returns filename of saved
		file
		"""
		data = ''.join(data)
		wf = wave.open((filename + '.wav'), 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(p.get_sample_size(self.FORMAT))
		wf.setframerate(self.RATE)  
		wf.writeframes(data)
		wf.close()

