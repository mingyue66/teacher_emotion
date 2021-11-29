import os

now_label = "/Users/mingyue/Desktop/application/WS/code/exp3_leave_one_speaker_out/label_new.txt"
old_label = "/Users/mingyue/Desktop/application/WS/code/exp3_leave_one_speaker_out/label_0727_identity.txt"


speaker_dic = {}
f2 = open(old_label,"r").readlines()
for line in f2:
	line = line.strip().split('\t')
	# print(line)
	idd = line[0]
	speaker = line[2]
	speaker_dic[idd] = speaker

# print(speaker_dic)

out_path = '/Users/mingyue/Desktop/application/WS/code/exp3_leave_one_speaker_out/label3_identity.txt'
out = open(out_path,"w")

cnt = 0
check = []
f1 = open(now_label,"r").readlines()
for line in f1:
	line = line.strip().split('\t')
	idd = line[0]
	emo = line[1]
	if idd not in speaker_dic:
		check.append(idd)
		new_line = idd + "\t" + emo + "\t" + "na"
	else:
		speaker = speaker_dic[idd]
		new_line = idd + "\t" + emo + "\t" + speaker
	out.write(new_line+'\n')
# print(cnt)

