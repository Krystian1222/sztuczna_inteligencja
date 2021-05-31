import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
m = 20
X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                             n_classes=2, n_samples=m, class_sep=10)
plt.scatter(X1[:, 0], X1[:, 1], marker = 'x', c=Y1, s=25, edgecolor='k')
Y1_kol = np.array([[-1 if y == 0 else y for y in Y1]]).T
kolumna_1 = np.ones((X1.shape[0], 1), dtype=X1.dtype)
X_train = np.hstack((kolumna_1, X1, Y1_kol))
k = 0
wk = []
eta = 0.5
rozmiar_w0 = X_train.shape[1] - 1
w0 = np.zeros((rozmiar_w0))
wk.append(w0)
while(True):
    E = [X for X in X_train if sum(X[:3] * w0) <= 0 and X[3] != -1 or sum(X[:3] * w0) > 0 and X[3] != 1]
    if len(E) == 0 or k >= 3000: break
    i = randint(0, len(E) - 1)
    yi = E[i][:3]
    xi = E[i][3]
    w0 = w0 + eta * yi * xi
    wk.append(w0)
    k += 1
wsp = wk[-1]
print("Współczynniki: {}, liczba iteracji: {}".format(wsp, k))
plt.plot(np.linspace(-14, 14, 50), -wsp[1]/wsp[2] * np.linspace(-14, 14, 50) - wsp[0]/wsp[2])
#plt.savefig("L09_Perceptron_Lebkuchen.png")
plt.show()