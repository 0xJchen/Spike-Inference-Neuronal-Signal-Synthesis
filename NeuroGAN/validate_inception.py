import pickle
import matplotlib.pyplot as plt
gen_dict = pickle.load(open("score", 'rb'))
is_score=gen_dict["is"]
print(is_score)
plt.plot(is_score)
plt.savefig("ISSCORE")
plt.close()
