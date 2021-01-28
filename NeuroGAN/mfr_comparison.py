import pickleshare
from oasis.oasis_methods import oasisAR1, oasisAR2
from oasis.plotting import simpleaxis
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
import numpy as np
import matplotlib.pyplot as plt
from sys import path
import pickle
from tqdm import trange
from collections import Counter
import pandas as pd
import collections
import scipy.stats as stats
import utility
path.append('..')

def gen_cal_stim(gan,i):
    return gan[i-1].T
def ori_cal_stim(data,label,i):
    idx=np.where(label==i)[0][1]
    return data[idx].reshape(29, 259).T
gan = np.asarray(pickle.load(open("stimulus", 'rb')))

print("generated signal: ",gan.shape)
#calcium under 2nd stimulus
print("generated calcium under 2nd stimulus ",gan[1].T.shape)
stim_1=gan[1].T#in fact, stimulus2

#load original signal
data = np.genfromtxt('../../20180827_Mouse1_reshape.csv', delimiter=',')
labels = np.genfromtxt('../../20180904_tag.csv', delimiter=',')

stimulus_id=8

ori_stim_1 = ori_cal_stim(data,labels,stimulus_id)
gen_stim_1 = gen_cal_stim(gan, stimulus_id)

# ori_stim_1 = data[40].reshape(29, 259)
# ori_stim_1 = ori_stim_1.T
# plt.close()

#intuition on cal
for neuron_id in range(30,40):
    plt.plot(ori_stim_1[neuron_id],label="original")
    plt.plot(gen_stim_1[neuron_id], label="generated")
    plt.legend()
    plt.savefig("./intuition/intuition-calcium"+str(neuron_id))
    plt.close()

#intuition on spike
neuron_id=1
for neuron_id in range(30,40):
    c, s, b, g, lam = deconvolve(gen_stim_1[neuron_id], penalty=1)
    plt.plot(s,label="gen")
    c, s, b, g, lam = deconvolve(ori_stim_1[neuron_id], penalty=1)
    plt.plot(s,label="ori")
    plt.legend()
    plt.savefig("./intuition/intuition-spike"+str(neuron_id))
    plt.close()

# plot spike counts for all neurons
new_fire_rate = []
old_fire_rate = []
new_spike_train=np.zeros((259,29))
old_spike_train=np.zeros((259,29))
for i in trange(259):
    c, s, b, g, lam = deconvolve(gen_stim_1[i], penalty=1)
    count=len(np.flatnonzero(s>0))
    new_fire_rate.append(count)
    new_spike_train[i]=s
    c, s, b, g, lam = deconvolve(ori_stim_1[i], penalty=1)
    count = len(np.flatnonzero(s > 0.5))
    old_fire_rate.append(count)
    old_spike_train[i]=s

#plot spike train
plt.matshow(old_spike_train)
plt.savefig("old_spike_train")
plt.close()
plt.matshow(new_spike_train)
plt.savefig("new_spike_train")
plt.close()

#generate kl-div for each neuron
def normalize(spike_train):
    if sum(spike_train)!=0:
        return [float(i)/sum(spike_train) for i in spike_train]
    else:
        return [1]


# old_spike_train=normalize(old_spike_train)
# new_spike_train = normalize(new_spike_train)
spike_kv_dis=[]
for i in range(259):
    norm_old = normalize(old_spike_train[i])
    norm_new=normalize(new_spike_train[i])
    if norm_old==[1] or norm_new == [1]:
        continue
    haha = stats.entropy(norm_old, norm_new)
    # print(haha)
    spike_kv_dis.append(haha)

spike_cnt = Counter(spike_kv_dis)

# print("**len: ",len(old_fire_rate))

plt.plot(np.asarray(old_fire_rate)/2.9,label="original")
plt.plot(np.asarray(new_fire_rate)/2.9, label="generated")
plt.xlabel("neuron id")
plt.ylabel("mean firing rate (Hz)")
plt.legend()
plt.savefig("compare-fire-rate")
plt.close()

ori_data = [int(i) for i in old_fire_rate]
gen_data=[int(i) for i in new_fire_rate]

ori_cnt=Counter(ori_data)
gen_cnt=Counter(gen_data)

def append_dict(dict):
    dict = {int(k): int(v) for k, v in dict.items()}
    for i in range(25):
        if (i) not in list(dict.keys()):
            dict[(i)]=0
    dict = collections.OrderedDict(sorted(dict.items()))
    return dict
ori_cnt=append_dict(ori_cnt)
gen_cnt=append_dict(gen_cnt)
# gen_cnt=utility.reshape_dict(gen_cnt)
#plot firing rate for all 259 neurons under one stimulus

y=list(ori_cnt.values())
x=list(ori_cnt.keys())
plt.bar(x,y,alpha=0.7,label="recorded signal")

y = list(gen_cnt.values())
x = list(gen_cnt.keys())
plt.bar(x,y,alpha=0.7,label="generated signal")
plt.xlabel("Firing rate (Hz)")
plt.ylabel("Neuron counts")
plt.legend()
plt.title("Firing Pattern under stimulus "+str(stimulus_id))
plt.savefig("compare-hz-stimulus"+str(stimulus_id))
plt.close()

#train svm for  mixed data
#preprocess




