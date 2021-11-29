import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import random
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

np.random.seed(3)
random.seed(3)

# read in text/acoustic data
# 两类ft的sample顺序完全一样
text_data = pd.read_csv('text_FT.csv',delimiter=',',header=0).to_numpy()
acoustic_data = pd.read_csv('FT_0726.csv',delimiter=',', header=None).to_numpy()

FT_text = text_data[:,2:]
FT_acoustic = acoustic_data[:,2:]

wav_id = text_data[:,0]
label = text_data[:,1]
encoder = LabelEncoder()
label = encoder.fit_transform(label)

#print(FT_acoustic.shape,FT_text.shape,label.shape)


cv = 5 # k-fold
w1_list = []
for i in range(0, 100, 1):
	w1_list.append(i * 1.0 / 100)

for w1 in w1_list:

	np.random.seed(3)
	random.seed(3)
	# train 2 models

	w2 = (1-w1)

	n_samples = int(FT_text.shape[0] / cv)
	all_indexes = list(range(FT_text.shape[0]))
	random.shuffle(all_indexes)

	clf_text = MLPClassifier(hidden_layer_sizes=(3),solver='lbfgs')
	clf_acoustic = MLPClassifier(hidden_layer_sizes=(3), solver='lbfgs')

	accs = []
	f1s = []
	clf1_accs=[]
	clf1_f1s=[]
	clf2_accs=[]
	clf2_f1s=[]

	for i in range(cv):
		if i == cv-1:
			test_indexes = list(range(i*n_samples,FT_text.shape[0]))
		else:
			test_indexes = list(range(i*n_samples,(i+1)*n_samples))
		train_indexes = [item for item in all_indexes if item not in test_indexes]

		# text model
		X_t_train = FT_text[train_indexes]
		y_t_train = label[train_indexes]
		X_t_test = FT_text[test_indexes]

		clf_text.fit(X_t_train, y_t_train)
		y_t_pred = clf_text.predict(X_t_test) 
		y_t_pred_proba = clf_text.predict_proba(X_t_test) 

		# acoutic model, needs std,PCA
		X_a_train = FT_acoustic[train_indexes]
		y_a_train = label[train_indexes] # 同上，一样的吧
		X_a_test = FT_acoustic[test_indexes]

		scaler = StandardScaler()
		X_a_train_std = scaler.fit_transform(X_a_train)
		X_a_test_std = scaler.transform(X_a_test)

		pca = PCA(n_components=0.9)
		X_a_train_std_pca = pca.fit_transform(X_a_train_std)
		X_a_test_std_pca = pca.transform(X_a_test_std)

		clf_acoustic.fit(X_a_train_std_pca, y_a_train)
		y_a_pred = clf_acoustic.predict(X_a_test_std_pca) 
		y_a_pred_proba = clf_acoustic.predict_proba(X_a_test_std_pca) 

		# print(np.sum(y_t_pred_proba))
		# print(np.sum(y_a_pred_proba))

		# 两个独立模型的performance
		y_test = label[test_indexes]

		acc_clf1 = np.sum(y_t_pred == y_test) * 1.0 / y_test.shape[0]
		f1_clf1 = f1_score(y_test,y_t_pred)
		clf1_accs.append(acc_clf1)
		clf1_f1s.append(f1_clf1)
		# print("Text model acc:{}  f1:{}".format('%.3f%%' % (acc_clf1 * 100),'%.3f%%' % (f1_clf1 * 100)))

		acc_clf2 = np.sum(y_a_pred == y_test) * 1.0 / y_test.shape[0]
		f1_clf2 = f1_score(y_test,y_a_pred)
		clf2_accs.append(acc_clf2)
		clf2_f1s.append(f1_clf2)
		# print("Acou model acc:{}  f1:{}".format('%.3f%%' % (acc_clf2 * 100),'%.3f%%' % (f1_clf2 * 100)))


		# 用得到的两个(class=1)proba线性组合
		t = (y_t_pred_proba[:,1]).reshape(-1,1)
		a = (y_a_pred_proba[:,1]).reshape(-1,1)

		p = w2*t + w1*a

		for i in p:
			index = np.argwhere(p==i)

			if i >= 0.5:
				p[index] = 1 
			else:
				p[index] = 0
		# print(y_test)

		y_pred = np.squeeze(p)
		# print(y_pred)
		# scores
		acc = np.sum(y_pred == y_test)*1.0 / y_test.shape[0]
		accs.append(acc)
		
		f1 = f1_score(y_test,y_pred)
		f1s.append(f1)	

		# print("linear  model acc:{}  f1:{}".format('%.3f%%' % (acc * 100),'%.3f%%' % (f1 * 100)))
		# print('--------------------------------------------')
		
		
	print('w1={}'.format(w1))	
	print("Text   model mean acc:{} mean f1 :{}".format('%.3f%%' % np.mean(clf1_accs),'%.3f%%' % np.mean(clf1_f1s)))
	print("Acou   model mean acc:{} mean f1 :{}".format('%.3f%%' % np.mean(clf2_accs),'%.3f%%' % np.mean(clf2_f1s)))

	print("linear model mean acc:{} mean f1 :{}".format('%.3f%%' % np.mean(accs),'%.3f%%' % np.mean(f1s)))

