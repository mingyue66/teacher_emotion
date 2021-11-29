# 处理6373个feature

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

np.random.seed(3)

# read-in
data = pd.read_csv('FT_3.csv', delimiter=',', header=None)
data = data.to_numpy()

wav_id = data[:,0] # 如果排序不变，未来应该能索引回来？
label = data[:,1]
FT = data[:,2:]
le = LabelEncoder()
label = le.fit_transform(label) 


# train_test_split # 用这种方法就索引不回来了
# X_train, X_test, y_train, y_test = train_test_split(FT, label, test_size=0.3)

shuffle_indexes = np.random.permutation(len(FT))
#print(shuffle_indexes)
#按什么比例分割
test_ratio = 0.3
#测试集的大小
test_size = int(test_ratio * len(FT))
#测试集的索引
test_indexes = shuffle_indexes[:test_size]
#训练集的索引
train_indexes = shuffle_indexes[test_size:]

X_test = FT[test_indexes]
X_train = FT[train_indexes]
y_test = label[test_indexes]
y_train = label[train_indexes]

#print(X_test.shape,y_train.shape)
#print(test_indexes,y_test)

# standardization
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.9)
X_train_std_pca = pca.fit_transform(X_train_std)
X_test_std_pca = pca.transform(X_test_std)

# print(X_train_std_pca.shape, X_test_std_pca.shape) # 6374 -- 140

# print(X_train, X_train_std, X_train_std_pca)
