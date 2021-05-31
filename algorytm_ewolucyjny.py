import numpy as np
import random
import matplotlib.pyplot as plt

class Evolutionary:
    def __init__(self, n = 5, pop = 10, gen_max = 10000, pc = 0.7, pm = 0.2):
        self.n = n
        self.pop = pop
        self.gen_max = gen_max
        self.pc = pc
        self.pm = pm
        self.ffmax = 0
        self.P = np.zeros((self.pop, self.n), int)
        self.Pn = np.zeros((self.pop, self.n), int)
        self.best = []
        self.avg = []

    def generate_population(self):
        for i in range(self.pop):
            self.P[i] = np.random.permutation(self.n)

    def evaluate(self):
        lista_kolizji = []
        i = 0
        while i < self.pop:
            lista_kolizji.append(self.collision_counts(self.P[i]))
            i += 1
        return lista_kolizji

    def solve(self):
        self.generate_population()
        gen = 0
        best = np.argmin(self.evaluate())
        while(gen < self.gen_max and self.collision_counts(self.P[best]) > self.ffmax):
            self.selection()
            self.crossover()
            self.mutation()
            x = self.evaluate()
            best = np.argmin(x)
            self.best.append(self.collision_counts(self.P[best]))
            self.avg.append(np.mean(x))
            gen += 1
        self.wykresy()
        self.wypiszSzachownice(self.P[best], self.n, True, True)
        return self.P[best], self.collision_counts(self.P[best])

    def wykresy(self):
        plt.figure(figsize=(10, 7))
        plt.plot(np.linspace(0, len(self.best) - 1, len(self.best)), self.best)
        plt.xlabel("Generacja")
        plt.ylabel("Przystosowanie")
        plt.title("Wykres zmienności wartości funkcji przystosowania najlepszego osobnika")
        plt.savefig("EA_Lebkuchen_przystosowanie_best.png")
        plt.figure(figsize=(10, 7))
        plt.plot(np.linspace(0, len(self.avg) - 1, len(self.avg)), self.avg)
        plt.xlabel("Generacja")
        plt.ylabel("Przystosowanie")
        plt.title("Wykres średniej wartości funkcji przystosowania z danej populacji")
        plt.savefig("EA_Lebkuchen_przystosowanie_mean.png")
        plt.show()

    def wypiszSzachownice(self, listaHetmanow, rozmiar, postacMacierzowa=True, postacTekstowa=False):
        if (postacTekstowa):
            for i in range(rozmiar):
                print('x: {}, y: {}'.format(i, listaHetmanow[i]))
            print()
        if (postacMacierzowa):
            for i in range(rozmiar):
                for j in range(rozmiar):
                    print('0 ', end='') if listaHetmanow[j] != i else print('1 ', end='')
                print()

    def mutate(self, indeks):
        i1 = random.randint(0, self.n - 1)
        i2 = random.randint(0, self.n - 1)
        self.P[indeks, i1], self.P[indeks, i2] = self.P[indeks, i2], self.P[indeks, i1]

    def collision_counts(self, listaHetmanow):
        liczba_bic = 0
        for i in range(self.n):
            for j in range(self.n):
                if (i != j and ((abs(listaHetmanow[i] - listaHetmanow[j]) == abs(i - j)) or (listaHetmanow[i] == listaHetmanow[j]))):
                    liczba_bic += 1
        return liczba_bic

    def selection(self):
        i = 0
        while i < self.pop:
            i1 = random.randint(0, self.pop-1)
            i2 = random.randint(0, self.pop-1)
            i1_kolizje = self.collision_counts(self.P[i1])
            i2_kolizje = self.collision_counts(self.P[i2])
            if(i1 != i2):
                self.Pn[i] = self.P[i1] if(i1_kolizje <= i2_kolizje) else self.P[i2]
                i += 1
        self.P = self.Pn

    def crossover(self):
        i = 0
        while(i < self.pop - 2):
            if(random.random() <= self.pc):
                self.cross(i, i + 1)
            i += 2

    def mutation(self):
        i = 0
        while i < self.pop:
            if(random.random() <= self.pm):
                self.mutate(i)
            i += 1

    def cross(self, indeks1, indeks2):
        s1 = dict()
        s2 = dict()
        maska = np.array([0 for i in range(self.n)])
        podzial = int(self.n / 3)
        maska[0:podzial] = 1
        np.random.shuffle(maska)
        for i in range(self.n):
            if maska[i]:
                s1[self.P[indeks1, i]] = self.P[indeks2, i]
                s2[self.P[indeks2, i]] = self.P[indeks1, i]
                self.P[indeks1, i], self.P[indeks2, i] = self.P[indeks2, i], self.P[indeks1, i]

        s1_klucze = s1.keys()
        s2_klucze = s2.keys()
        for i in range(self.n):
            if not maska[i]:
                if self.P[indeks2, i] in s1_klucze:
                    self.P[indeks2, i] = s1[self.P[indeks2, i]]
                if self.P[indeks1, i] in s2_klucze:
                    self.P[indeks1, i] = s2[self.P[indeks1, i]]

if __name__ == '__main__':
    AG = Evolutionary(n=8)
    naj, liczba_bic = AG.solve()
    print("Najlepszy osobnik:", naj)
    print("Liczba bić:", liczba_bic)