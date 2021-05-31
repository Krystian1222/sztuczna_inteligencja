from statistics import mean, stdev
import math
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NBC_dyskretny(BaseEstimator, ClassifierMixin):
    def __init__(self, poprawka_laplace = False):
        self.lista_pstwo_Y = []
        self.lista_pstwo_Y_licznosci = []
        self.lista_klas = []
        self.dane_na_klase = {}
        self.slownik_pstwo_X_pod_war_Y = {}
        self.poprawka_laplace = poprawka_laplace

    def fit(self, X, y):
        liczba_klas = len(self.lista_klas)
        rozmiar_y = y.size
        (self.lista_klas, self.lista_pstwo_Y_licznosci) = np.unique(y, return_counts = True)
        self.lista_pstwo_Y = (self.lista_pstwo_Y_licznosci + 1) / (rozmiar_y + liczba_klas) if self.poprawka_laplace else self.lista_pstwo_Y_licznosci / rozmiar_y
        self.dane_na_klase = self.fit_dane_na_klase(X, y, rozmiar_y)
        self.slownik_pstwo_X_pod_war_Y = self.fit_pstwo_X_war_Y(X, y)

    def fit_dane_na_klase(self, X, y, rozmiar_y):
        for yi in self.lista_klas:
            self.dane_na_klase[yi] = np.array([X[i] for i in range(rozmiar_y) if y[i] == yi])
        return self.dane_na_klase

    def fit_pstwo_X_war_Y(self, X, y):
        liczba_kolumn_X = X.shape[1]
        X_unikalne = np.unique(X)
        liczba_klas = len(self.lista_klas)
        y = 0
        while y < liczba_klas:
            slownik_pstwo_X_pod_war_Y = {}
            i = 0
            while i < liczba_kolumn_X:
                slownik_pstwo_X_pod_war_Y[i] = {j:self.P_X_war_Y_Laplace(X, y, i, j) for j in X_unikalne}
                i += 1
            self.slownik_pstwo_X_pod_war_Y[y] = slownik_pstwo_X_pod_war_Y
            y += 1
        return self.slownik_pstwo_X_pod_war_Y

    def P_X_war_Y_Laplace(self, X, y, i, j):
        dane_na_klase = self.dane_na_klase[y][:, i] == j
        if self.poprawka_laplace:
            unikalne_X = np.unique(X).shape[0]
            licznik = np.count_nonzero(dane_na_klase) + 1
            mianownik = self.lista_pstwo_Y_licznosci[y] + unikalne_X
        else:
            licznik = np.count_nonzero(dane_na_klase)
            mianownik = self.lista_pstwo_Y_licznosci[y]
        return licznik / mianownik

    def predict(self, X):
        liczba_klas = len(self.lista_klas)
        liczba_wierszy = X.shape[0]
        liczba_kolumn = X.shape[1]
        x = 0
        y_gwiazdka = []
        while x < liczba_wierszy:
            lista_1 = np.ones((liczba_klas))
            k = 0
            while k < liczba_klas:
                y = 0
                while y < liczba_kolumn:
                    lista_1[k] *= self.slownik_pstwo_X_pod_war_Y[k][y][X[x, y]] * self.lista_pstwo_Y[k]
                    y += 1
                k += 1
            y_gwiazdka.append(np.argmax(lista_1))
            x += 1
        return np.array(y_gwiazdka)

    def predict_proba(self, X):
        liczba_klas = len(self.lista_klas)
        liczba_wierszy = X.shape[0]
        liczba_kolumn = X.shape[1]
        x = 0
        y_gwiazdka = []
        while x < liczba_wierszy:
            lista_1 = np.ones((liczba_klas))
            k = 0
            while k < liczba_klas:
                y = 0
                while y < liczba_kolumn:
                    lista_1[k] *= self.slownik_pstwo_X_pod_war_Y[k][y][X[x, y]] * self.lista_pstwo_Y[k]
                    y += 1
                k += 1
            p = 0
            lista_2 = []
            while p < liczba_klas:
                el = lista_1[p] / sum(lista_1)
                lista_2.append(el)
                y_gwiazdka.append(lista_2)
                p += 1
            x += 1

        return np.array(y_gwiazdka)

class NBC_ciagly(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.lista_pstwo_Y = []
        self.lista_klas = []
        self.lista_pstwo_Y_licznosci = []
        self.dane_na_klase = {}
        self.slownik_pstwo_X_pod_war_Y = {}

    def fit(self, X, y):
        self.lista_klas, self.lista_pstwo_Y = np.unique(y, return_counts = True)
        rozmiarY = y.size
        liczba_klas = len(self.lista_klas)
        y_i = 0
        while y_i < liczba_klas:
            self.dane_na_klase[y_i] = np.array([X[i] for i in range(rozmiarY) if y_i == y[i]])
            y_i += 1

    def predict_petla(self, X):
        liczba_wierszy = X.shape[0]
        liczba_kolumn = X.shape[1]
        liczba_klas = len(self.lista_klas)
        macierz_1 = np.ones((liczba_wierszy, liczba_klas))
        k = 0
        while k < liczba_klas:
            y = 0
            while y < liczba_kolumn:
                j = 0
                while j < liczba_wierszy:
                    mianownik_podstawy = mean(self.dane_na_klase[k][:, y]) * math.sqrt(2 * math.pi)
                    licznik_wykladnika = (X[j, y] - mean(self.dane_na_klase[k][:, y])) ** 2
                    mianownik_wykladnika = 2 * (stdev(self.dane_na_klase[k][:, y]) ** 2)
                    macierz_1[j, k] *= (1 / mianownik_podstawy) * math.e ** (-licznik_wykladnika/mianownik_wykladnika)
                    j += 1
                y += 1
            k += 1
        return macierz_1

    def predict(self, X):
        liczba_wierszy = X.shape[0]
        macierz_wynik = self.predict_petla(X)
        return [np.argmax(macierz_wynik[i]) for i in range(liczba_wierszy)]

    def predict_proba(self, X):
        liczba_wierszy = X.shape[0]
        liczba_klas = len(self.lista_klas)
        macierz_wynik = self.predict_petla(X)
        return np.array([[(macierz_wynik[i, j] / sum(macierz_wynik[i])) for j in range(liczba_klas)] for i in range(liczba_wierszy)])

def dokladnosc(y_sklas, y_prawdziwe):
    licznik = 0
    rozmiar_Y = y_prawdziwe.size
    for i in range(rozmiar_Y):
        if(y_prawdziwe[i] == y_sklas[i]):
            licznik += 1
    return (licznik / rozmiar_Y) * 100

def dokladnosc_PP(y_sklas, y_prawdziwe, liczba_klas):
    licznik = 0
    rozmiarY = len(y_prawdziwe)
    for i in range(rozmiarY):
        for j in range(liczba_klas):
            if(y_prawdziwe[i][j] == y_sklas[i][j]):
                licznik += 1
    return (licznik / (rozmiarY * liczba_klas)) * 100

if __name__ == "__main__":
    zbior_wina = datasets.load_wine()
    zbior_wina_dane = zbior_wina.data
    zbior_wina_klasy = zbior_wina.target

    encode = 'ordinal'
    strategy = 'uniform'
    n_bins = 3
    dyskretyzacja = KBinsDiscretizer(
        n_bins = n_bins,
        encode = encode,
        strategy = strategy
    )

    dyskretyzacja.fit(zbior_wina_dane)
    zbior_wina_dane_B = dyskretyzacja.transform(zbior_wina_dane)

    # dane zdyskretyzowane
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(zbior_wina_dane_B, zbior_wina_klasy)

    # dane ciagle
    X_train, X_test, y_train, y_test = train_test_split(zbior_wina_dane, zbior_wina_klasy)

    nbc_dyskretny = NBC_dyskretny(poprawka_laplace=False)
    nbc_dyskretny.fit(X_train_B, y_train_B)
    y_P_B = nbc_dyskretny.predict(X_test_B)
    y_PP_B = nbc_dyskretny.predict_proba(X_test_B)

    nbc_ciagly = NBC_ciagly()
    nbc_ciagly.fit(X_train, y_train)
    y_P_C = nbc_ciagly.predict(X_test)
    y_PP_C = nbc_ciagly.predict_proba(X_test)

    nbc_wbudowany = GaussianNB()
    nbc_wbudowany.fit(zbior_wina_dane, zbior_wina_klasy)
    #
    y_P_BI = nbc_wbudowany.predict(X_test_B)
    y_P_C_BI = nbc_wbudowany.predict(X_test)
    y_PP_BI = nbc_wbudowany.predict_proba(X_test_B)
    y_PP_C_BI = nbc_wbudowany.predict_proba(X_test)
    #
    liczba_klas = len(np.unique(zbior_wina_klasy))
    print("Dokładność: Predict zdyskretyzowany: {}%".format(dokladnosc(y_P_B, y_P_BI)))
    print("Dokładność: Predict proba zdyskretyzowany: {}%".format(dokladnosc_PP(y_PP_B, y_PP_BI, liczba_klas)))
    print("Dokładność: Predict ciągły: {}%".format(dokladnosc(y_P_C, y_test)))
    print("Dokładność: Predict proba ciągły: {}%".format(dokladnosc_PP(y_PP_C, y_PP_BI, liczba_klas)))
    # print("Predict zdyskretyzowany: ", end = '')
    # print("Równy") if(y_P_BI == y_P_B).all else print("Różny")
    # print("Predict ciągły: ", end = '')
    # print("Równy") if(y_P_C == y_P_C_BI).all else print("Różny")
    # print("Predict proba zdyskretyzowany: ", end = '')
    # print("Równy") if(y_PP_BI == y_PP_B) else print("Różny")
    # print("Predict proba ciągły: ", end = '')
    # print("Równy") if(y_PP_C == y_PP_C_BI).all else print("Różny")
    print("Predict proba ciągły:")
    print(y_PP_C)

    print("Predict proba ciągły wbudowany:")
    print(y_PP_BI)