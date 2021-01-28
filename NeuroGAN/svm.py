import pickleshare
# from oasis.oasis_methods import oasisAR1, oasisAR2
# from oasis.plotting import simpleaxis
# from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
import numpy as np
import matplotlib.pyplot as plt
from sys import path
import pickle
from tqdm import trange
from torchvision import transforms
from collections import Counter
import pandas as pd
import collections
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# from loader import *
# from model import *
path.append('..')
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
scaler = MinMaxScaler(feature_range=(0, 1))
data = np.genfromtxt('20180827_Mouse1_reshape.csv', delimiter=',')
labels = np.genfromtxt('20180904_tag.csv', delimiter=',')
labels-=1
# print(scaler.fit(data))
data = data.clip(50, 1000)
#498*7511 0,1
# data = scaler.transform(data)
#498*7511 -1,1
# data = img_transform(data)
data = scale(data)
print(data.shape)
recorded_data = data.reshape(498,29*259)
recorded_label = labels[:498]
print("record mean: ", np.asarray(recorded_data).mean())
print("record data len: ",len(recorded_data), len(recorded_label))

#load gen data
gen_dict = (pickle.load(open("stimuli_svm", 'rb')))
gen_data=np.asarray(gen_dict["data"])
gen_label=np.asarray(gen_dict["label"])
gen_data = gen_data.reshape(112, 29*259)
print("loaded gen data: ",gen_data.shape)


# gen_scaler = MinMaxScaler(feature_range=(-1, 1))
# print(gen_scaler.fit(gen_data))
# gen_scaler = gen_scaler.transform(gen_data)
gen_data=scale(gen_data)
# # gen_data = img_transform(gen_data)
# gen_data = gen_data.reshape(112, 29*259)

print("gen after trnasform: ",gen_data.shape,gen_label.shape,gen_data.mean())

plt.matshow(gen_data)
plt.savefig("gen data")
plt.close()
# total_data=np.concatenate((gen_data,recorded_data))
# total_label=np.concatenate((gen_label,recorded_label))
total_data = recorded_data
total_label = recorded_label
X_train, X_test, y_train, y_test = train_test_split(
    total_data, total_label, test_size=0.1, train_size=0.9, random_state=10)
non_linear_model = SVC(kernel='rbf')
non_linear_model.fit(X_train, y_train)
y_pred = non_linear_model.predict(X_test)
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

