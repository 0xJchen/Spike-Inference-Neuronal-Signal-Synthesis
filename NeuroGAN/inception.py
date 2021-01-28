# define batch=200 to compute IS
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# scaling the features
from sklearn.preprocessing import scale
from tqdm import trange
import scipy.stats as stats
from validation import generate_svm_data
import matplotlib.pyplot as plt
# assumes images have the shape 299x299x3, pixels in [0,255]
epoch_info = int(sys.argv[1])
# generate data during batch interval
klscore = []
isscore = []
for j in trange(epoch_info):
    generate_svm_data(j)
    # load gen_data
    stimuli_svm = "stimuli_svm_"+str(j)
    gen_dict = pickle.load(open(stimuli_svm, 'rb'))
    gen_data = gen_dict["data"]
    gen_label = gen_dict["label"]
    gen_data = np.asarray(gen_data)
    gen_data = gen_data.reshape(112, 29*259)
    # gen_data:112,29*259

    # load ori_data
    ori_data = np.genfromtxt('20180827_Mouse1_reshape.csv', delimiter=',')
    ori_label = np.genfromtxt('20180827_Mouse1_tag.csv', delimiter=',')
    ori_label = ori_label[:498]
    # print(ori_label[470:500])
    ori_label -= 1
    # ori_data: 498,29*259

    # scale_data
    print(gen_data.shape, ori_data.shape)

    gen_data = scale(gen_data)
    ori_data = scale(ori_data)
    print("label*****: ", np.asarray(ori_label).max(),
          np.asarray(gen_label).max())
    # concatenate
    data = np.concatenate((gen_data, ori_data))
    label = np.concatenate((gen_label, ori_label))

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.1, train_size=0.9, random_state=10)

    # define SVM
    non_linear_model = SVC(kernel='linear', probability=True)

    # fit
    non_linear_model.fit(X_train, y_train)

    # predict
    y_pred = non_linear_model.predict(X_test)

    # accuracy
    print("accuracy:", metrics.accuracy_score(
        y_true=y_test, y_pred=y_pred), "\n")

    non_linear_model.probability = True
    print("*****test x: ", X_test)
    # shape:61*14
    proba = non_linear_model.predict_proba(X_test)
    y_hat = np.mean(proba, axis=0)
    print(y_hat.shape, y_hat)

    eps = 1E-16
    kl = []
    for i in range(proba.shape[0]):
        cur_kl = stats.entropy(y_hat+eps, proba[i]+eps)
        kl.append(cur_kl)
    mean_kl = np.asarray(kl).mean()
    print("mean kl", mean_kl)
    print("IS score: ", np.exp(mean_kl))
    isscore.append(np.exp(mean_kl))
    klscore.append(mean_kl)
plt.plot(isscore)
plt.savefig("ISscore")
plt.close()
score = {}
score["kl"] = klscore
score["is"] = isscore
pickle.dump(score, open("score", "wb"))

gen_dict = pickle.load(open("score", 'rb'))
is_score = gen_dict["is"]
print(is_score)
plt.plot(is_score)
plt.savefig("ISSCORE")
plt.close()
