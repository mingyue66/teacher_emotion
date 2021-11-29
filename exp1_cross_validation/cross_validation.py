import numpy as np 
import random
np.random.seed(3)

## read in data
# import sys
# sys.path.insert(0, '/Users/mingyue/Desktop/application/WS/code/00_preprocessing_audio_label')
from process_feature import label, FT
X = FT 
y = label


## build models
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.svm import SVC
svc = SVC()

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(hidden_layer_sizes=(3), solver='lbfgs')


## cross validation
print("------------ start cv -------------")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score

cv = 5 	# 5-folds cross validation
n_samples = int(X.shape[0] / cv) 	# sample numbers in each fold
all_indexes = list(range(X.shape[0]))
random.shuffle(all_indexes)

accs = []
f1s = []
ps = []
rs = []
for i in range(cv):

	## train_test split
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
	scaler = StandardScaler()
	X_train_std = scaler.fit_transform(X_train)
	X_test_std = scaler.transform(X_test)

	## PCA
	pca = PCA(n_components=0.9)
	X_train_std_pca = pca.fit_transform(X_train_std)
	X_test_std_pca = pca.transform(X_test_std)

	print(X_test_std_pca.shape)

	## fit different models
	svc.fit(X_train_std_pca, y_train) 		
	y_pred = svc.predict(X_test_std_pca)

	print(y_pred)

	## scores
	acc = np.sum(y_pred == y_test)*1.0 / y_test.shape[0]
	accs.append(acc)

	f1 = f1_score(y_test,y_pred)
	f1s.append(f1)

	p = precision_score(y_test,y_pred)
	r = recall_score(y_test,y_pred)
	ps.append(p)
	rs.append(r)
	# print(acc, f1)


## return mean scores
print("------------ scores -------------")
print("acc {} mean {}".format(accs,np.mean(accs)))
print("f1 {} mean {}".format(f1s,np.mean(f1s)))
print("precision {} mean {}".format(ps,np.mean(ps)))
print("recall {} mean {}".format(rs,np.mean(rs)))
