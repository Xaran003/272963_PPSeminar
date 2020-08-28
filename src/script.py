"""Data utworzenia: 23.07.2020"""
"""Adrian Dzwonkowski, index: 272963, Przedmiot: Kurs Python"""

"zakomentować mdshare z powodu otrzymywanego błedu wgrania danych"
import mdshare
import numpy as np, matplotlib.pyplot as plt, argparse
from sklearn.manifold import TSNE

"Wczytywanie danych. Występowały problemy z załadowaniem ich ze źródła, dlatego też za Pana radą zostały " \
"one wgrane bezpośrednio z komputera, nalezy odkomentować dataset1 i zakomentować drugie"
#dataset = 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz'  #bezpośrednie wczytywanie
dataset = mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-distances.npz')  #wgrywanie ze źródła
with np.load(dataset) as f:
    X = np.vstack([f[key] for key in sorted(f.keys())])

"Funkcja argparse wymagana do pracy z okna komend. Zastosowanie odpowiednich flag"
parser = argparse.ArgumentParser(description = "Projekt programu redukcji wymiarowości danych.")
parser.add_argument('-s', '--step', metavar = '', type = int, default = 750, help =
"Ustawienie kroku wczytywania danych. Im wyższy parametr  tym więcej próbek zostanie zignorowanych. Bazowo 750")
parser.add_argument('-ds', '--dot_size', metavar = '', type = int, default = 20, help =
"Rozmiar punktow na wykresie (Bazowo 20).")
parser.add_argument('-x', '--x_scale', metavar = '', type = float, default = None, help=
"wyswietla wykres w przedziale od -a do a (w osi x )")
parser.add_argument('-y', '--y_scale', metavar = '', type = float, default = None, help=
"wyswietla wykres w przedziale od -a do a (w osi y )")
parser.add_argument('-a', '--alpha', metavar = '', type = float, default = 1, help =
"od 0 do 1; 0 - niewidoczne punkty; 1- nieprzezroczyste punkty (Bazowo 1).")
# "metavar = ''  --> usuwa opis zmiennej w pomocy po komendach (np. -n , --name; zamiast: -n NAME, --name NAME)"

args = parser.parse_args()
dot_size = args.dot_size
step = args.step
v_alpha = args.alpha
xs = args.x_scale
ys = args.y_scale

"Rzutowanie wielowymiarowego tensora na trójwymiarową przestrzeń, gdzie brana jest pod uwagę każda próbka"
Y = TSNE(n_components=3).fit_transform(X[::step])

"Zmniejszenie wymiarowości wykonywane poprzez moduł t-SNE"
# Skalowanie 1-wymiaru tablicy 2-wymiarowej. Dopasowanie do zakresu od -pi do pi
Y[:, 0] = np.interp(Y[:, 0], (Y[:, 0].min(), Y[:, 0].max()), (-np.pi, np.pi))
# Skalowanie 2-wymiaru tablicy 2-wymiarowej. Dopasowanie do zakresu od -pi do pi
Y[:, 1] = np.interp(Y[:, 1], (Y[:, 1].min(), Y[:, 1].max()), (-np.pi, np.pi))
# Skalowanie danych dla dopasowania do zakresu od -pi do pi
Y[:, 2] = np.interp(Y[:, 2], (Y[:, 2].min(), Y[:, 2].max()), (-np.pi, np.pi))

"Wykonywanie wykresu 'plot'"
# Wykres punktowy. Pierwsza kolumna x, drugą y i trzecia z (ustawienia koloru).
# Pod tym linkiem są cmap (kolory) - https://matplotlib.org/examples/color/colormaps_reference.html
plt.scatter(Y[:, 0], Y[:, 1], c=Y[:, 2], s=dot_size, alpha=v_alpha, cmap='magma',)
if xs is not None:
    plt.xlim(-np.pi, np.pi)   # Ustawienie zakresu dla osi x od -pi do pi
plt.xticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) # Zakres etykiety osi x od od -pi do pi
if ys is not None:
    plt.ylim(-np.pi, np.pi)   # Ustawienie zakresu dla osi y od -pi do pi
plt.yticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) # Zakres etykiety osi y od od -pi do pi

plt.axis('scaled') # Równe wyskalowanie osi wykresu
cbar = plt.colorbar()     # Rysowanie kolumny zakresu koloru na wykresie
cbar.set_ticks([-np.pi, 0, np.pi]) # Ustawienie zakresu kolumny koloru od -pi do pi
cbar.set_ticklabels(['-π', 0, 'π']) #Ustawienie napisu zakresu od -pi do pi
plt.show()  # Wywołanie wykresu


