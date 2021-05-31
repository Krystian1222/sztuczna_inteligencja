import time
import matplotlib.pyplot as plt
from queue import PriorityQueue

class N_Hetmanow:
    def __init__(self, N):
        self.N = N
        self.wyniki = open('wynikiBrutePoprawione_PQ.csv', encoding='utf-8', mode='w')

    def inicjalizacja(self, N):
        i = 0
        lH = []
        while i < N:
            lH.append(-1)
            i = i + 1
        return lH

    def liczNiePuste(self, lH, N):
        S = 0
        for i in range(N):
            if (lH[i] != -1):
                S += 1
        return S

    def wstawHetmana(self, X, Y, lH, N):
        if (lH[X] == -1):
            for i in range(N):
                if (lH[i] != -1):
                    if (abs(lH[i] - Y) == abs(i - X)):
                        return False

                if (Y == lH[i]): return False
                if (i == lH[X]): return False

            lH[X] = Y
            return True
        return False

    def metodaBruteForce(self, lH, N):
        bruteForceLista = []
        for i in range(N):
            for j in range(N):
                nLH = lH[:]
                if (self.wstawHetmana(i, j, nLH, N)):
                    bruteForceLista.append(nLH)
        return bruteForceLista

    def separacjaWKolumnach(self, lH, N):
        sL = []
        i = 0
        while (i < N):
            if (lH[i] == -1):
                break
            i = i + 1
        for j in range(N):
            nLH = lH[:]
            if (self.wstawHetmana(i, j, nLH, N)):
                sL.append(nLH)
        return sL

    def wypiszSzachownice(self, lH, N, macierz=True, tekst=False):
        if (tekst):
            for i in range(N):
                print('x: {}, y: {}'.format(i, lH[i]))
            print()
        if (macierz):
            for i in range(N):
                for j in range(N):
                    print('0 ', end='') if lH[i] != j else print('1 ', end='')
                print()

    def petlaBFS(self, N, kol, sS, sW, bruteForce = False, macierz=True, tekst=False):
        while (len(kol) > 0):
            aS = kol[0][:]
            del kol[0]
            if (self.liczNiePuste(aS, N) == N):
                self.wypiszSzachownice(aS, N, macierz, tekst)
                break
            if (bruteForce): nS = self.metodaBruteForce(aS, N)
            else: nS = self.separacjaWKolumnach(aS, N)
            sS += 1
            dlugosc = len(nS)
            k = 0
            while(k < dlugosc):
                kol.append(nS[k])
                sW = sW + 1
                k = k + 1
        return sW, sS

    def BFS(self, N, bruteForce=False, macierz=True, tekst=False):
        sS = 0
        sW = 0
        start = time.time()
        kol = []
        kol.append(self.inicjalizacja(N))
        sW, sS = self.petlaBFS(N, kol, sS, sW, bruteForce, macierz, tekst)
        end = time.time()
        cO = end - start
        return cO, sW, sS

    def petlaDFS(self, N, kol, sW, sS, bruteForce=False, macierz=True, tekst=False):
        while (len(kol) > 0):
            aS = kol[len(kol) - 1][:]
            del kol[len(kol) - 1]
            if (self.liczNiePuste(aS, N) == N):
                self.wypiszSzachownice(aS, N, macierz, tekst)
                break
            if (bruteForce): nS = self.metodaBruteForce(aS, N)
            else: nS = self.separacjaWKolumnach(aS, N)
            sS = sS + 1
            x = 0
            dlugosc = len(nS)
            while(x < dlugosc):
                kol.append(nS[x])
                sW = sW + 1
                x = x + 1
        return sW, sS

    def DFS(self, N, bruteForce=False, macierz=True, tekst=False):
        sS = 0
        sW = 0
        start = time.time()
        kol = []
        kol.append(self.inicjalizacja(N))
        sW, sS = self.petlaDFS(N, kol, sW, sS, bruteForce, macierz, tekst)
        end = time.time()
        cO = end - start
        return cO, sW, sS

    def H1(self, row, N):
        S = 0
        i = 0
        while(i < N):
            if(row[i] != -1):
                if(row[i] < N / 2): S += N - row[i] + 1
                else: S += row[i]
            i = i + 1
        return (N - self.liczNiePuste(row, N)) * S

    def H2(self, lH, N):
        S = 0
        i = 0
        j = 0
        while(i < N):
            while(j < N):
                if(self.wstawHetmana(i, j, lH, N)): S += 1
                j = j + 1
            i = i + 1
        return pow(N, 2) - S
    def H3(self, lH, N):
        dH = 0
        k = 0
        l = 0
        while k < N:
            if lH[k] != -1:
                while l < N:
                    if lH[l] != -1:
                        if lH[k] != lH[l]:
                            dH += 1
                    l = l + 1
            k = k + 1
        S = ((1 + N - 1)/2)*(N-1)
        return (S - dH)

    def petlaBestFirstSearch(self, N, kol, sS, sW, heurystyka, bruteForce=False, macierz=True, tekst=False):
        while(not kol.empty()):
            sS += 1
            aS = kol.get()
            #print("aktualny stan")
            #print(aS)
            #print(N)
            if (self.liczNiePuste(aS[1], N) == N):
                self.wypiszSzachownice(aS[1], N, macierz, tekst)
                break
            if (bruteForce): nS = self.metodaBruteForce(aS[1], N)
            else: nS = self.separacjaWKolumnach(aS[1], N)

            if (heurystyka == 1):
                for i in range(len(nS)):
                    kol.put(((self.H1(nS[i], N)), nS[i]))
                    sW += 1
            if(heurystyka == 2):
                for i in range(len(nS)):
                    kol.put(((self.H2(nS[i], N)), nS[i]))
                    sW += 1
            if (heurystyka == 3):
                for i in range(len(nS)):
                    kol.put(((self.H3(nS[i], N)), nS[i]))
                    sW += 1
        return sW, sS
    def BestFirstSearch(self, N, heurystyka, bruteForce=False, macierz=True, tekst=False):
        sS = 0
        sW = 0
        start = time.time()
        kol = PriorityQueue()
        kol.put((1000, self.inicjalizacja(N)))
        #print(kol.get())
        sW, sS = self.petlaBestFirstSearch(N, kol, sS, sW, heurystyka, bruteForce, macierz, tekst)
        end = time.time()
        cO = end - start
        return cO, sW, sS

    def zapisDoPliku(self):
        self.wyniki.write('algorytm,strategia,heurystyka,rozmiar szachownicy,stany wygenerowane,stany sprawdzone,czas operacji (sekundy)\n')

    def liczBFS(self, N, bruteForce=True):
        czasBFS = []
        stanyWBFS = []
        stanySBFS = []
        osX = []
        for i in range(4, N):
            print(i)
            czOp, stW, stS = self.BFS(i, bruteForce, True, False)
            if bruteForce: self.wyniki.write("BFS,BruteForce,brak,{},{},{},{:.6f}\n".format(i, stW, stS, czOp))
            else: self.wyniki.write("BFS,Ulepszona,brak,{},{},{},{:.6f}\n".format(i, stW, stS, czOp))
            osX.append(i)
            stanyWBFS.append(stW)
            stanySBFS.append(stS)
            czasBFS.append(czOp)
        return osX, czasBFS, stanyWBFS, stanySBFS

    def liczDFS(self, N, bruteForce=True):
        czasDFS = []
        stanyWDFS = []
        stanySDFS = []
        osX = []
        for i in range(4, N):
            print(i)
            czOp, stW, stS = self.DFS(i, bruteForce, True, False)
            if bruteForce: self.wyniki.write("DFS,BruteForce,brak,{},{},{},{:.6f}\n".format(i, stW, stS, czOp))
            else: self.wyniki.write("DFS,Ulepszona,brak,{},{},{},{:.6f}\n".format(i, stW, stS, czOp))
            osX.append(i)
            stanyWDFS.append(stW)
            stanySDFS.append(stS)
            czasDFS.append(czOp)
        return osX, czasDFS, stanyWDFS, stanySDFS

    def liczBestFirstSearch(self, N, H, bruteForce = False):
        czasBestFS = []
        stanyWBestFS = []
        stanySBestFS = []
        osX = []
        for i in range(4, N):
            print(i)
            czOp, stW, stS = self.BestFirstSearch(i, H, bruteForce, True, False)
            if bruteForce: self.wyniki.write("BestFirstSearch,BruteForce,{},{},{},{},{:.6f}\n".format(H, i, stW, stS, czOp))
            else: self.wyniki.write("BestFirstSearch,Ulepszona,{},{},{},{},{:.6f}\n".format(H, i, stW, stS, czOp))
            osX.append(i)
            stanyWBestFS.append(stW)
            stanySBestFS.append(stS)
            czasBestFS.append(czOp)
        return osX, czasBestFS, stanyWBestFS, stanySBestFS

    def wykresySmart(self):
        osX, czasBFS, stanyWBFS, stanySBFS = self.liczBFS(self.N, bruteForce=False)
        osX, czasDFS, stanyWDFS, stanySDFS = self.liczDFS(self.N, bruteForce=False)
        osX1, czasBestFS1, stanyWBestFS1, stanySBestFS1 = self.liczBestFirstSearch(self.N, 1, bruteForce=False)
        osX2, czasBestFS2, stanyWBestFS2, stanySBestFS2 = self.liczBestFirstSearch(self.N, 2, bruteForce=False)
        osX3, czasBestFS3, stanyWBestFS3, stanySBestFS3 = self.liczBestFirstSearch(self.N, 3, bruteForce=False)
        plt.figure(figsize=(8, 6))
        plt.title("1.2.2 Ulepszona reprezentacja (separacja w kolumnach)")
        plt.xlabel('Wartość n')
        plt.ylabel('Czas wykonania [s]')
        plt.xticks(osX)
        plt.plot(osX, czasBFS, 'r-o', label='BFS')
        plt.plot(osX, czasDFS, 'b-o', label='DFS')
        plt.plot(osX1, czasBestFS1, 'g-o', label='BestFS, H1')
        plt.plot(osX2, czasBestFS2, 'm-o', label='BestFS, H2')
        plt.plot(osX3, czasBestFS3, 'y-o', label='BestFS, H3')
        plt.legend(loc='upper left')
        plt.savefig("CzasyWykonania122_{}.png".format(self.N - 1))
        plt.figure(figsize=(8, 6))
        plt.xticks(osX)
        plt.title("1.2.2 Ulepszona reprezentacja (separacja w kolumnach)")
        plt.xlabel('Wartość n')
        plt.ylabel('Liczba stanów wygenerowanych')
        plt.plot(osX, stanyWBFS, 'r-o', label='BFS')
        plt.plot(osX, stanyWDFS, 'b-o', label='DFS')
        plt.plot(osX1, stanyWBestFS1, 'g-o', label='BestFS, H1')
        plt.plot(osX2, stanyWBestFS2, 'm-o', label='BestFS, H2')
        plt.plot(osX3, stanyWBestFS3, 'y-o', label='BestFS, H3')
        plt.legend(loc='upper left')
        plt.savefig("StanyWygenerowane122_{}.png".format(self.N - 1))
        plt.figure(figsize=(8, 6))
        plt.xticks(osX)
        plt.title("1.2.2 Ulepszona reprezentacja (separacja w kolumnach)")
        plt.xlabel('Wartość n')
        plt.ylabel('Liczba stanów sprawdzonych')
        plt.plot(osX, stanySBFS, 'r-o', label='BFS')
        plt.plot(osX, stanySDFS, 'b-o', label='DFS')
        plt.plot(osX1, stanySBestFS1, 'g-o', label='BestFS, H1')
        plt.plot(osX2, stanySBestFS2, 'm-o', label='BestFS, H2')
        plt.plot(osX3, stanySBestFS3, 'y-o', label='BestFS, H3')
        plt.legend(loc='upper left')
        plt.savefig("StanySprawdzone122_{}.png".format(self.N - 1))

    def wykresyBrute(self):
        osX, czasBFS, stanyWBFS, stanySBFS = self.liczBFS(self.N, bruteForce=True)
        osX, czasDFS, stanyWDFS, stanySDFS = self.liczDFS(self.N, bruteForce=True)
        osX1, czasBestFS1, stanyWBestFS1, stanySBestFS1 = self.liczBestFirstSearch(self.N, 1, bruteForce=True)
        osX2, czasBestFS2, stanyWBestFS2, stanySBestFS2 = self.liczBestFirstSearch(self.N, 2, bruteForce=True)
        osX3, czasBestFS3, stanyWBestFS3, stanySBestFS3 = self.liczBestFirstSearch(self.N, 3, bruteForce=True)
        plt.figure(figsize=(8, 6))
        plt.title("1.2.1 Podejście pierwsze (najprostsze)")
        plt.xlabel('Wartość n')
        plt.ylabel('Czas wykonania [s]')
        plt.xticks(osX)
        plt.plot(osX, czasBFS, 'r-o', label='BFS')
        plt.plot(osX, czasDFS, 'b-o', label='DFS')
        plt.plot(osX1, czasBestFS1, 'g-o', label='BestFS, H1')
        plt.plot(osX2, czasBestFS2, 'm-o', label='BestFS, H2')
        plt.plot(osX3, czasBestFS3, 'y-o', label='BestFS, H3')
        plt.legend(loc='upper left')
        plt.savefig("CzasyWykonania121_{}.png".format(self.N - 1))
        plt.figure(figsize=(8, 6))
        plt.xticks(osX)
        plt.title("1.2.1 Podejście pierwsze (najprostsze)")
        plt.xlabel('Wartość n')
        plt.ylabel('Liczba stanów wygenerowanych')
        plt.plot(osX, stanyWBFS, 'r-o', label='BFS')
        plt.plot(osX, stanyWDFS, 'b-o', label='DFS')
        plt.plot(osX1, stanyWBestFS1, 'g-o', label='BestFS, H1')
        plt.plot(osX2, stanyWBestFS2, 'm-o', label='BestFS, H2')
        plt.plot(osX3, stanyWBestFS3, 'y-o', label='BestFS, H3')
        plt.legend(loc='upper left')
        plt.savefig("StanyWygenerowane121_{}.png".format(self.N - 1))
        plt.figure(figsize=(8, 6))
        plt.xticks(osX)
        plt.title("1.2.1 Podejście pierwsze (najprostsze)")
        plt.xlabel('Wartość n')
        plt.ylabel('Liczba stanów sprawdzonych')
        plt.plot(osX, stanySBFS, 'r-o', label='BFS')
        plt.plot(osX, stanySDFS, 'b-o', label='DFS')
        plt.plot(osX1, stanySBestFS1, 'g-o', label='BestFS, H1')
        plt.plot(osX2, stanySBestFS2, 'm-o', label='BestFS, H2')
        plt.plot(osX3, stanySBestFS3, 'y-o', label='BestFS, H3')
        plt.legend(loc='upper left')
        plt.savefig("StanySprawdzone121_{}.png".format(self.N - 1))

    def wyswietlWykresyzamknijPlik(self):
        plt.show()
        self.wyniki.close()


n_hetmanow = N_Hetmanow(8)
#n_hetmanow.BestFirstSearch(5, 2, False, True, False)
n_hetmanow.zapisDoPliku()
n_hetmanow.wykresyBrute()
n_hetmanow.wyswietlWykresyzamknijPlik()
# if __name__ == '__main__':
#     # rozmiar = 7
#     # n_hetmanow = N_Hetmanow(rozmiar + 1)
#     # n_hetmanow.zapisDoPliku()
#     # n_hetmanow.wykresyBrute()
#     n_hetmanow = N_Hetmanow(13)
#     n_hetmanow.rozmiar = 13
#     n_hetmanow.zapisDoPliku()
#     n_hetmanow.wykresyBrute()
#     n_hetmanow.wyswietlWykresyzamknijPlik()
# else:
#     pass
#

