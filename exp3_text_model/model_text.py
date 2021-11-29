import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import random
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

np.random.seed(3)

data = pd.read_csv('text_FT.csv',header=0)
data = data.to_numpy()
X = data[:,2:]
label = data[:,1]

encoder = LabelEncoder()
y = encoder.fit_transform(label)

print(encoder.classes_)

# train clf model

cv = 5 # k-fold
n_samples = int(X.shape[0] / cv)
all_indexes = list(range(X.shape[0]))
random.shuffle(all_indexes)

ann = MLPClassifier(hidden_layer_sizes=(3), solver='lbfgs')
gnb = GaussianNB()

accs = []
f1s = []
for i in range(cv):
	if i == cv-1:
		test_indexes = list(range(i*n_samples,X.shape[0]))
	else:
		test_indexes = list(range(i*n_samples,(i+1)*n_samples))
	train_indexes = [item for item in all_indexes if item not in set(test_indexes)]

	X_train = X[train_indexes]
	y_train = y[train_indexes]
	X_test = X[test_indexes]
	y_test = y[test_indexes]

	## standardization
	# scaler = StandardScaler()
	# X_train_std = scaler.fit_transform(X_train)
	# X_test_std = scaler.transform(X_test)

	# ## PCA
	# pca = PCA(n_components=0.9)
	# X_train_std_pca = pca.fit_transform(X_train_std)
	# X_test_std_pca = pca.transform(X_test_std)

	ann.fit(X_train, y_train)
	y_pred = ann.predict(X_test)
	print(y_pred)

	# scores
	acc = np.sum(y_pred == y_test)*1.0 / y_test.shape[0]
	accs.append(acc)

	f1 = f1_score(y_test,y_pred)
	f1s.append(f1)

	# Locate badcases

	# 整个testing set里的索引
	false_postives = []
	false_negatives = []

	for i in range(y_test.shape[0]):
		true = y_test[i]
		pred = y_pred[i]

		if true == 0 and pred == 1:
			false_postives.append(i)
		if true == 1 and pred == 0:
			false_negatives.append(i)
	#print(false_postives, false_negatives)



	from process_feature import test_indexes, wav_id
	for num in false_postives:

		# 在打乱顺序前，在整个 dataset的索引
		test_index = test_indexes[num]
		# 在wav对应关系
		wav = wav_id[test_index]
		# print(wav)

	for num in false_negatives:

		# 在打乱顺序前，在整个 dataset的索引
		test_index = test_indexes[num]
		# 在wav对应关系
		wav = wav_id[test_index]
		print(wav)


	

	
print(accs,np.mean(accs))
print(f1s,np.mean(f1s))

