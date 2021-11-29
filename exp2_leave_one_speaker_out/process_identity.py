feature_file = "/Users/mingyue/Desktop/application/WS/code/exp2_leave_one_speaker_out/FT_3.csv"
identity_file = "/Users/mingyue/Desktop/application/WS/code/exp2_leave_one_speaker_out/label3_identity.txt"

# get the index of all id, and the corresponding emo label

indexes = []
dic = {}
ff = open(feature_file,"r").readlines()
for line in ff:
	line = line.strip().split(',')
	wav_id = line[0]
	label = line[1]
	indexes.append(wav_id)
	dic[wav_id] = label
#print(indexes.index('242-022219s-3-40_255.40'))

# get the ids of each speaker

for i in range(12):
	locals()['speaker_wav_'+str(i)] = []

idf = open(identity_file,"r").readlines()
for line in idf:
	line = line.strip().split('\t')
	for i in range(12):
	    
		if int(line[2]) == i:
			locals()['speaker_wav_'+str(i)].append(line[0]) 

# get the indexes of each speaker 

for i in range(12):
	locals()['speaker_index_'+str(i)] = []

for i in range(12):
	for item in locals()['speaker_wav_'+str(i)]:
		#print(indexes.index(item))
		locals()['speaker_index_'+str(i)].append(indexes.index(item))

for i in range(12):
	# print("{}  Speaker_index_{}:{}".format(len(locals()['speaker_index_'+str(i)]),i,locals()['speaker_index_'+str(i)]))
	print("speaker_index_{}={}".format(i,locals()['speaker_index_'+str(i)]))


# get the number of emo labels of each speaker 
to = 0
for i in range(12):
	p_cnt = 0
	n_cnt = 0
	for wav in locals()['speaker_wav_'+str(i)]:
		l = dic[wav]
		if l == 'pos':
			p_cnt += 1
		
		elif l == 'neg':
			n_cnt += 1
		to += 1
	print("Speaker{}: pos {} neg {} ".format(i,p_cnt,n_cnt))



