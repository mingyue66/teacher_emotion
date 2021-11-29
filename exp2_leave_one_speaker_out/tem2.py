import os
import pandas

FT_path = '/Users/mingyue/Desktop/application/WS/code/exp2_leave_one_speaker_out/FT_3.csv'
identity_path = '/Users/mingyue/Desktop/application/WS/code/exp2_leave_one_speaker_out/label3_identity.txt'

speaker0_wav = []
fi = open(identity_path,"r").readlines()
for line in fi:
	line = line.strip().split('\t')
	wav_id = line[0]
	speaker = line[2]
	if speaker == '11':
		speaker0_wav.append(wav_id)
print(speaker0_wav)


li = []
ft = pandas.read_csv(FT_path, delimiter=',', header=None)
for i in range(ft.shape[0]):
	if ft[0][i] in speaker0_wav:
		li.append(i)

print(li)