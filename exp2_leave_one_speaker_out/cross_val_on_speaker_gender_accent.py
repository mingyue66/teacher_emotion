# test generalizability on all speakers

import numpy as np 
import random
np.random.seed(3)

# read in data
from process_feature import label, FT

X = FT 
y = label
all_indexes = list(range(X.shape[0])) # cannot shuffle

speaker_index_0=[14, 40, 87, 150, 161, 176, 186, 226, 259, 294, 300]
speaker_index_1=[5, 7, 18, 24, 25, 34, 36, 38, 39, 42, 54, 61, 64, 66, 67, 69, 71, 72, 75, 80, 86, 88, 90, 98, 101, 102, 103, 106, 107, 110, 115, 116, 119, 123, 124, 125, 128, 129, 130, 133, 135, 137, 138, 143, 151, 156, 163, 164, 165, 166, 168, 170, 172, 174, 175, 177, 182, 185, 193, 194, 197, 199, 200, 201, 202, 203, 205, 232, 233, 235, 237, 239, 243, 246, 250, 255, 257, 258, 262, 267, 270, 271, 274, 280, 281, 286, 287, 290, 291, 293, 295, 297, 298, 309, 315, 316, 329, 330, 332, 335, 346, 350, 355, 357]
speaker_index_2=[26, 43, 47, 83, 93, 100, 141, 198, 209, 221, 261, 272, 289, 306, 311, 321, 340, 343]
speaker_index_3=[3, 41, 117, 127, 146, 160, 171, 183, 248, 276, 345, 358]
speaker_index_4=[6, 9, 28, 32, 121, 157, 179, 210, 223, 228, 236, 245, 303, 325, 352]
speaker_index_5=[0, 1, 2, 10, 17, 19, 35, 45, 49, 50, 52, 53, 56, 58, 68, 76, 81, 89, 95, 99, 105, 109, 112, 114, 118, 134, 139, 148, 152, 154, 167, 173, 180, 187, 191, 208, 211, 214, 218, 219, 222, 224, 225, 229, 234, 240, 242, 251, 254, 256, 260, 264, 265, 288, 292, 307, 310, 313, 319, 320, 326, 338, 344, 347, 349, 351, 353, 354, 356]
speaker_index_6=[21, 92, 97, 247, 268]
speaker_index_7=[12, 77, 82, 111, 162, 204, 206, 231, 238, 241, 263, 296, 336]
speaker_index_8=[33, 37, 44, 48, 55, 57, 73, 84, 91, 96, 108, 120, 136, 145, 181, 192, 220, 227, 230, 244, 273, 277, 278, 285, 302, 308, 314, 323, 324, 327, 339, 341]
speaker_index_9=[15, 20, 27, 30, 78, 79, 85, 126, 132, 140, 144, 169, 178, 184, 195, 213, 216, 252, 269, 275, 305, 318]
speaker_index_10=[29, 51, 94, 147, 188, 190, 212, 217, 266, 283, 304, 312, 317, 348]
speaker_index_11=[4, 8, 11, 13, 16, 22, 23, 31, 46, 59, 60, 62, 63, 65, 70, 74, 104, 113, 122, 131, 142, 149, 153, 155, 158, 159, 189, 196, 207, 215, 249, 253, 279, 282, 284, 299, 301, 322, 328, 331, 333, 334, 337, 342, 359, 360]


# k-fold validation (leave one speaker out)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,roc_auc_score

n_speaker = 12 

from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(hidden_layer_sizes=(3), solver='lbfgs')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.svm import SVC
svc = SVC()

accs = []
f1s = []
aucs = []

for i in range(n_speaker):

	test_indexes = locals()['speaker_index_'+str(i)]
	train_indexes = [item for item in all_indexes if item not in set(test_indexes)]

	X_train = X[train_indexes]
	y_train = y[train_indexes]
	X_test = X[test_indexes]
	y_test = y[test_indexes]

	scaler = StandardScaler()
	X_train_std = scaler.fit_transform(X_train)
	X_test_std = scaler.transform(X_test)

	## PCA
	pca = PCA(n_components=0.9)
	X_train_std_pca = pca.fit_transform(X_train_std)
	X_test_std_pca = pca.transform(X_test_std)

	svc.fit(X_train_std_pca, y_train)
	y_pred = svc.predict(X_test_std_pca)

	# scores
	acc = np.sum(y_pred == y_test)*1.0 / y_test.shape[0]
	accs.append(acc)
	
	f1 = f1_score(y_test,y_pred)
	f1s.append(f1)


print("------------ scores -------------")
print("acc {} mean {}".format(accs,np.mean(accs)))
print("f1 {} mean {}".format(f1s,np.mean(f1s)))



