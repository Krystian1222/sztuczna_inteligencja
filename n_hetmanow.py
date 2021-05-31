import time
import matplotlib.pyplot as plt

class N_Hetmanow:
    def __init__(self, rozmiar):
        self.rozmiar = rozmiar
        self.wyniki = open('wynikiOO.csv', encoding='utf-8', mode='w')

    def liczNiePuste(self, listaHetmanow, rozmiar):
        suma = 0
        for i in range(rozmiar):
            if (listaHetmanow[i] != -1):
                suma += 1
        return suma

    def wstawHetmana(self, pozX, pozY, listaHetmanow, rozmiar):
        if (listaHetmanow[pozX] == -1):
            for i in range(rozmiar):
                if (listaHetmanow[i] != -1):
                    if (abs(listaHetmanow[i] - pozY) == abs(i - pozX)):
                        return False

                if (pozY == listaHetmanow[i]):
                    return False

                if (i == listaHetmanow[pozX]):
                    return False

            listaHetmanow[pozX] = pozY
            return True
        return False

    def metodaBruteForce(self, listaHetmanow, rozmiar):
        bruteForceLista = []
        for i in range(rozmiar):
            for j in range(rozmiar):
                nowaListaHetmanow = listaHetmanow[:]
                if (self.wstawHetmana(i, j, nowaListaHetmanow, rozmiar)):
                    bruteForceLista.append(nowaListaHetmanow)
        return bruteForceLista

    def separacjaWKolumnach(self, listaHetmanow, rozmiar):
        separacjaLista = []
        i = 0
        while (i < rozmiar):
            if (listaHetmanow[i] == -1):
                break
            i += 1
        for j in range(rozmiar):
            nowaListaHetmanow = listaHetmanow[:]
            if (self.wstawHetmana(i, j, nowaListaHetmanow, rozmiar)):
                separacjaLista.append(nowaListaHetmanow)
        return separacjaLista

    def wypiszSzachownice(self, listaHetmanow, rozmiar, postacMacierzowa=True, postacTekstowa=False):
        if (postacTekstowa):
            for i in range(rozmiar):
                print('x: {}, y: {}'.format(i, listaHetmanow[i]))
            print()
        if (postacMacierzowa):
            for i in range(rozmiar):
                for j in range(rozmiar):
                    print('0 ', end='') if listaHetmanow[i] != j else print('1 ', end='')
                print()

    def BFS(self, rozmiar, bruteForce=False, postacMacierzowa=True, postacTekstowa=False):
        stanySprawdzone = 0
        stanyWygenerowane = 0
        start = time.time()
        kolejka = []
        kolejka.append([-1 for i in range(self.rozmiar)])
        while (len(kolejka) > 0):
            aktualnyStan = kolejka[0][:]
            del kolejka[0]
            if (self.liczNiePuste(aktualnyStan, rozmiar) == rozmiar):
                self.wypiszSzachownice(aktualnyStan, rozmiar, postacMacierzowa, postacTekstowa)
                break
            if (bruteForce):
                noweStany = self.metodaBruteForce(aktualnyStan, rozmiar)
            else:
                noweStany = self.separacjaWKolumnach(aktualnyStan, rozmiar)
            stanySprawdzone += 1
            for i in range(len(noweStany)):
                kolejka.append(noweStany[i])
                stanyWygenerowane += 1
        end = time.time()
        czasOperacji = end - start
        return czasOperacji, stanyWygenerowane, stanySprawdzone

    def DFS(self, rozmiar, bruteForce=False, postacMacierzowa=True, postacTekstowa=False):
        stanySprawdzone = 0
        stanyWygenerowane = 0
        start = time.time()
        kolejka = []
        kolejka.append([-1 for i in range(self.rozmiar)])
        while (len(kolejka) > 0):
            aktualnyStan = kolejka[len(kolejka) - 1][:]
            del kolejka[len(kolejka) - 1]
            if (self.liczNiePuste(aktualnyStan, rozmiar) == rozmiar):
                self.wypiszSzachownice(aktualnyStan, rozmiar, postacMacierzowa, postacTekstowa)
                break
            if (bruteForce):
                noweStany = self.metodaBruteForce(aktualnyStan, rozmiar)
            else:
                noweStany = self.separacjaWKolumnach(aktualnyStan, rozmiar)

            stanySprawdzone += 1

            for i in range(len(noweStany)):
                kolejka.append(noweStany[i])
                stanyWygenerowane += 1
        end = time.time()
        czasOperacji = end - start
        return czasOperacji, stanyWygenerowane, stanySprawdzone

    def wypiszSzachownice(self, listaHetmanow, rozmiar, postacMacierzowa=True, postacTekstowa=False):
        if (postacTekstowa):
            for i in range(rozmiar):
                print('x: {}, y: {}'.format(i, listaHetmanow[i]))
            print()
        if (postacMacierzowa):
            for i in range(rozmiar):
                for j in range(rozmiar):
                    print('0 ', end='') if listaHetmanow[i] != j else print('1 ', end='')
                print()

    def zapisDoPliku(self):
        self.wyniki.write('algorytm,strategia,rozmiar szachownicy,stany wygenerowane,stany sprawdzone,czas operacji (sekundy)\n')

    def liczBFS(self, rozmiar, bruteForce=True):
        czasBFS = []
        stanyWBFS = []
        stanySBFS = []
        osX = []
        for i in range(4, rozmiar):
            print(i)
            czOp, stW, stS = self.BFS(i, bruteForce, True, False)
            if bruteForce:
                self.wyniki.write("BFS,BruteForce,{},{},{},{:.6f}\n".format(i, stW, stS, czOp))
            else:
                self.wyniki.write("BFS,Ulepszona,{},{},{},{:.6f}\n".format(i, stW, stS, czOp))
            osX.append(i)
            stanyWBFS.append(stW)
            stanySBFS.append(stS)
            czasBFS.append(czOp)
        return osX, czasBFS, stanyWBFS, stanySBFS

    def liczDFS(self, rozmiar, bruteForce=True):
        czasDFS = []
        stanyWDFS = []
        stanySDFS = []
        osX = []
        for i in range(4, rozmiar):
            print(i)
            czOp, stW, stS = self.DFS(i, bruteForce, True, False)
            if bruteForce:
                self.wyniki.write("DFS,BruteForce,{},{},{},{:.6f}\n".format(i, stW, stS, czOp))
            else:
                self.wyniki.write("DFS,Ulepszona,{},{},{},{:.6f}\n".format(i, stW, stS, czOp))
            osX.append(i)
            stanyWDFS.append(stW)
            stanySDFS.append(stS)
            czasDFS.append(czOp)
        return osX, czasDFS, stanyWDFS, stanySDFS

    def wykresySmart(self):
        #osX, czasBFS, stanyWBFS, stanySBFS = self.liczBFS(self.rozmiar, bruteForce=False)
        osX, czasDFS, stanyWDFS, stanySDFS = self.liczDFS(self.rozmiar, bruteForce=False)
        plt.figure(figsize=(8, 6))
        plt.title("1.2.2 Ulepszona reprezentacja (separacja w kolumnach)")
        plt.xlabel('Wartość n')
        plt.ylabel('Czas wykonania [s]')
        plt.xticks(osX)
        #plt.plot(osX, czasBFS, 'ro', label='BFS')
        plt.plot(osX, czasDFS, 'bo', label='DFS')
        plt.legend(loc='upper left')
        plt.savefig("CzasyWykonania122samBFS{}.png".format(self.rozmiar - 1))
        plt.figure(figsize=(8, 6))
        plt.xticks(osX)
        plt.title("1.2.2 Ulepszona reprezentacja (separacja w kolumnach)")
        plt.xlabel('Wartość n')
        plt.ylabel('Liczba stanów wygenerowanych')
        #plt.plot(osX, stanyWBFS, 'ro', label='BFS')
        plt.plot(osX, stanyWDFS, 'bo', label='DFS')
        plt.legend(loc='upper left')
        plt.savefig("StanyWygenerowane122samBFS{}.png".format(self.rozmiar - 1))
        plt.figure(figsize=(8, 6))
        plt.xticks(osX)
        plt.title("1.2.2 Ulepszona reprezentacja (separacja w kolumnach)")
        plt.xlabel('Wartość n')
        plt.ylabel('Liczba stanów sprawdzonych')
        #plt.plot(osX, stanySBFS, 'ro', label='BFS')
        plt.plot(osX, stanySDFS, 'bo', label='DFS')
        plt.legend(loc='upper left')
        plt.savefig("StanySprawdzone122samBFS{}.png".format(self.rozmiar - 1))

    def wykresyBrute(self):
        #osX, czasBFS, stanyWBFS, stanySBFS = self.liczBFS(self.rozmiar, bruteForce=True)
        osX, czasDFS, stanyWDFS, stanySDFS = self.liczDFS(self.rozmiar, bruteForce=True)

        plt.figure(figsize=(8, 6))
        plt.title("1.2.1 Podejście pierwsze (najprostsze)")
        plt.xlabel('Wartość n')
        plt.ylabel('Czas wykonania [s]')
        plt.xticks(osX)
        #plt.plot(osX, czasBFS, 'ro', label='BFS')
        plt.plot(osX, czasDFS, 'bo', label='DFS')
        plt.legend(loc='upper left')
        plt.savefig("CzasyWykonania121_{}.png".format(self.rozmiar - 1))
        plt.figure(figsize=(8, 6))
        plt.xticks(osX)
        plt.title("1.2.1 Podejście pierwsze (najprostsze)")
        plt.xlabel('Wartość n')
        plt.ylabel('Liczba stanów wygenerowanych')
        #plt.plot(osX, stanyWBFS, 'ro', label='BFS')
        plt.plot(osX, stanyWDFS, 'bo', label='DFS')
        plt.legend(loc='upper left')
        plt.savefig("StanyWygenerowane121_{}.png".format(self.rozmiar - 1))
        plt.figure(figsize=(8, 6))
        plt.xticks(osX)
        plt.title("1.2.1 Podejście pierwsze (najprostsze)")
        plt.xlabel('Wartość n')
        plt.ylabel('Liczba stanów sprawdzonych')
        #plt.plot(osX, stanySBFS, 'ro', label='BFS')
        plt.plot(osX, stanySDFS, 'bo', label='DFS')
        plt.legend(loc='upper left')
        plt.savefig("StanySprawdzone121_{}.png".format(self.rozmiar - 1))

    def wyswietlWykresyzamknijPlik(self):
        plt.show()
        self.wyniki.close()

if __name__ == '__main__':
    # rozmiar = 7
    # n_hetmanow = N_Hetmanow(rozmiar + 1)
    # n_hetmanow.zapisDoPliku()
    # n_hetmanow.wykresyBrute()
    n_hetmanow = N_Hetmanow(13)
    n_hetmanow.rozmiar = 13
    n_hetmanow.zapisDoPliku()
    n_hetmanow.wykresyBrute()
    n_hetmanow.wyswietlWykresyzamknijPlik()
else:
    pass


