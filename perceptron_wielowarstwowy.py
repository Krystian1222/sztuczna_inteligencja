import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class MLP:
    def __init__(self, hidden = 100, epochs = 1000, eta = 0.01, shuffle = True):
        self.hidden = hidden
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.lista_kosztow = []

    def _sigmoid(self, z):
        return 1 / (1 + math.e ** (-z))

    def _forward(self, X):
        wyjscie_ukryta_warstwa = X.dot(self.w_h) + self.b_h
        f_akt_neur_ukryte = self._sigmoid(wyjscie_ukryta_warstwa)
        wyj_z_sum_war_wyj = f_akt_neur_ukryte.dot(self.w_out) + self.b_out
        f_akt_neur_wyj = self._sigmoid(wyj_z_sum_war_wyj)
        return f_akt_neur_ukryte, f_akt_neur_wyj

    def _compute_cost(self, y, output):
        return -np.sum((y * np.log(output)) + (1 - y) * np.log(1 - output))

    def fit(self, X_train, y_train, X_test, y_test, tytul):
        X_train_liczba_cech = X_train.shape[1]
        y_train_liczba_cech = y_train.shape[1]
        liczba_neuronow_ukr = self.hidden
        self.w_h = np.random.normal(loc = 0, scale = 0.1, size = (X_train_liczba_cech, liczba_neuronow_ukr))
        self.b_h = np.zeros(liczba_neuronow_ukr)
        self.w_out = np.random.normal(loc = 0, scale = 0.1, size = (liczba_neuronow_ukr, y_train_liczba_cech))
        self.b_out = np.zeros(y_train_liczba_cech)
        self.lista_y = []
        self.lista_acc = []
        self.lista_y_test = []
        self.lista_acc_test = []

        for i in range(self.epochs):
            if self.shuffle:
                zbior_uczacy = np.hstack((X_train, y_train))
                np.random.shuffle(zbior_uczacy)
                tmp, wyj_z_sieci = self._forward(X_train)
                self.lista_y.append(self._compute_cost(y_train, wyj_z_sieci))
            else:
                tmp, wyj_z_sieci = self._forward(X_train)
                self.lista_y.append(self._compute_cost(y_train, wyj_z_sieci))

            for j in zip(X_train, y_train):
                a_ukryte, a_out = self._forward(X_train)
                poch_FA_wej = a_out * (1 - a_out)
                delta_out = (a_out - y_train) * poch_FA_wej
                poch_FA_ukr = a_ukryte * (1 - a_ukryte)
                delta_h = np.dot(delta_out, self.w_out.T) * poch_FA_ukr
                grdnt_w_h = np.dot(X_train.T, delta_h)
                grdnt_b_h = delta_h
                grdnt_w_out = np.dot(a_ukryte.T, delta_out)
                grdnt_b_out = delta_out

                self.w_h -= (grdnt_w_h * self.eta)
                self.b_h -= (np.sum(grdnt_b_h) * self.eta)
                self.w_out -= (grdnt_w_out * self.eta)
                self.b_out -= (np.sum(grdnt_b_out) * self.eta)

            self.lista_kosztow.append(self._compute_cost(y_train, a_out))
            lista_pred = self.predict(X_train)
            lista_y_oryg = [np.argmax(y_train[i]) for i in range(len(y_train))]
            self.lista_acc.append(self.accuracy(lista_pred, lista_y_oryg))

            lista_pred_test = self.predict(X_test)
            lista_y_oryg_test = [np.argmax(y_test[i]) for i in range(len(y_test))]
            self.lista_acc_test.append(self.accuracy(lista_pred_test, lista_y_oryg_test))
        print("Dokładność dla zbioru testowego: {}%".format(sum(self.lista_acc_test) / len(self.lista_acc_test)))

        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, self.epochs, self.epochs), self.lista_y)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function", color = 'blue')
        ax2 = ax.twinx()
        ax2.plot(np.linspace(0, self.epochs, self.epochs), self.lista_acc, color = 'orange')
        plt.title(tytul)
        ax2.set_ylabel("Accuracy", color = 'orange')
        plt.savefig("MLP_Lebkuchen_" + tytul + ".png")

    def predict(self, X):
        tmp1, tmp2 = self._forward(X)
        return [np.argmax(tmp2[i]) for i in range(len(tmp2))]

    def accuracy(self, y1, y2):
        licznik = 0
        for i in range(len(y1)):
            if (y1[i] == y2[i]):
                licznik += 1
        return (licznik / len(y1)) * 100

def Zadanie1(zadanie_3 = False):
    print("Zadanie 1")
    zbior_irysow = datasets.load_iris()
    zbior_irysow_dane = zbior_irysow.data
    zbior_irysow_klasy = zbior_irysow.target
    zbior_irysow_klasy_wektor = np.zeros((zbior_irysow_klasy.size, 3))
    tytul = "Zadanie 1"
    if zadanie_3:
        scaler = MinMaxScaler()
        zbior_irysow_dane = scaler.fit_transform(zbior_irysow_dane)
        print("Normalizacja danych")
        tytul = "Zadanie 1 + Zadanie 3"

    for i in range(zbior_irysow_klasy.size):
        zbior_irysow_klasy_wektor[i, zbior_irysow_klasy[i]] = 1

    X_train, X_test, y_train, y_test = train_test_split(zbior_irysow_dane, zbior_irysow_klasy_wektor, random_state = 13)
    emelpe = MLP()
    emelpe.fit(X_train, y_train, X_test, y_test, tytul)

def Zadanie2(zadanie_3 = False):
    print("Zadanie 2")
    zbior_liczb = datasets.load_digits(2)
    zbior_liczb_dane = zbior_liczb.data
    zbior_liczb_klasy = zbior_liczb.target
    zbior_liczb_klasy_wektor = np.zeros((zbior_liczb_klasy.size, 2))
    tytul = "Zadanie 2"
    if zadanie_3:
        scaler = MinMaxScaler()
        zbior_liczb_dane = scaler.fit_transform(zbior_liczb_dane)
        print("Normalizacja danych")
        tytul = "Zadanie 2 + Zadanie 3"

    for i in range(zbior_liczb_klasy.size):
        zbior_liczb_klasy_wektor[i, zbior_liczb_klasy[i]] = 1

    X_train, X_test, y_train, y_test = train_test_split(zbior_liczb_dane, zbior_liczb_klasy_wektor, random_state=13)
    emelpe = MLP()
    emelpe.fit(X_train, y_train, X_test, y_test, tytul)

Zadanie1(zadanie_3=False)
Zadanie2(zadanie_3=False)
Zadanie1(zadanie_3=True)
Zadanie2(zadanie_3=True)
plt.show()

# Wykresy wygenerowano dla następujących danych:
# hidden = 100, epochs = 1000, eta = 0.01, shuffle = True

# Uczenie wsadowe
# Normalizacja poprawia efektywność o około 4 punkty procentowe dla zadania 1.
# Dla zadania 2 nieznacznie ją pogarsza o około 0.02 punktu procentowego.

# Przykładowe wyniki
# Zadanie 1
# Normalizacja: Nie. Dokładność dla zbioru testowego: 90.97368421052633%
# Normalizacja: Tak. Dokładność dla zbioru testowego: 94.81052631578841%
# Zadanie 2
# Normalizacja: Nie. Dokładność dla zbioru testowego: 99.87333333333333%
# Normalizacja: Tak. Dokładność dla zbioru testowego: 99.85444444444444%

# Uczenie on-line
# W zadaniu 1 normalizacja poprawia efektywność o około 2.3 punktu procentowego.
# Dla zadania 2 dokładność w obu przypadkach wynosi 100%.

# Przykładowe wyniki
# Zadanie 1
# Normalizacja: Nie. Dokładność dla zbioru testowego: 94.22631578947377%
# Normalizacja: Tak. Dokładność dla zbioru testowego: 96.55789473684152%
# Zadanie 2
# Normalizacja. Nie. Dokładność dla zbioru testowego: 100.0%
# Normalizacja. Tak. Dokładność dla zbioru testowego: 100.0%