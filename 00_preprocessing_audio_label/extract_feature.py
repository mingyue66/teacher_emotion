import os
import opensmile
import pandas as pd

path = '/Users/mingyue/Desktop/application/WS/code/00_preprocessing_audio_label/wav'
label_path = '/Users/mingyue/Desktop/application/WS/code/00_preprocessing_audio_label/labels/label_3.txt'


# read in id and labels
label = {}
f = open(label_path,'r').readlines()
for line in f:
    line = line.strip().split('\t')
    label[line[0]]=line[1]


# extract feature
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)


all_FT_list = []

for file in os.listdir(path):
    wav_id = file.split('.wav')[0]
    wav_label = label[wav_id]

    # only process pos & neg labels (for binary classification)
    if wav_label in ['pos','neg']:

        wav_path = os.path.join(path,file)
        FT = smile.process_file(wav_path)

        # insert wav id to the first col 
        col_name=FT.columns.tolist()
        col_name.insert(0,'wav_id')
        col_name.insert(1,'label')
        FT=FT.reindex(columns=col_name)
        FT['wav_id'] = [wav_id]
        FT['label'] = [wav_label]

        ft_list = FT.values.tolist()
        all_FT_list.append(ft_list[0])

    
all_FT = pd.DataFrame(all_FT_list)
all_FT.to_csv('FT_3.csv', sep=',', index=False, header=False, float_format='%.6f')


