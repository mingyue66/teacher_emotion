import string
import nltk
import numpy as np

text_file = 'transcription_tim.txt' # transcribed by Tim
emo_label_file = '../label_0726.txt' # labeled by Mingyue

# get emo labels
label_dic = {}
wav_ids = []
e = open(emo_label_file,'r').readlines()
for line in e:
	line = line.strip().split('\t')
	wav_ids.append(line[0])
	label_dic[line[0]]=line[1]


# get all words
text_dic = {}
all_words = ''
f = open(text_file,'r').readlines()
for line in f:
	line = line.strip().split('\t')
	wav_id = line[0]
	text = line[2]

	if wav_id in wav_ids:
		# remove punctuations 
		for punc in string.punctuation:
			if punc != '\'':  # e.g. we'll
				text = text.replace(punc, '')
		# lower case 
		text = text.lower()

		text_dic[wav_id] = text
		all_words += text + ' '
#print(all_words)

vocab = all_words.split() # 2447
print(len(vocab))


## optional：去掉低频词 ##
vocab_selected = []
for word in vocab:
	if vocab.count(word) > 1:
		vocab_selected.append(word)

print(len(vocab_selected))
## 去掉低频词 ##

vocab_selected = list(set(vocab_selected)) # 573
print(len(vocab_selected))

#print(vocab)

# text to vector
FT = np.zeros(shape=(len(wav_ids),len(vocab_selected))) #(343, 573)
labels = []

for i,wav in enumerate(wav_ids):
	# FT
	text = text_dic[wav]
	text = text.strip().split()
	for word in text:
		if word in vocab_selected:
			index = vocab_selected.index(word)
			FT[i,index] += 1

	# label
	emo_label = label_dic[wav]
	if emo_label == "pos":
		labels.append('pos')
	elif emo_label == "neg":
		labels.append('neg')
	else:
		print("wrong!")

label = np.array(labels) # (343,)


# # export to csv
iid = np.array(wav_ids).reshape(-1,1)
lab = label.reshape(-1,1)
data = np.concatenate((lab,FT), axis=1)
data = np.concatenate((iid,data), axis=1)

import pandas as pd 
vocab_selected.insert(0,'label')
vocab_selected.insert(0,'wav_id')
pd.DataFrame(data).to_csv("text_FT.csv", header=vocab_selected, index=False)









