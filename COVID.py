# clustering_analysis

# pobierz dane covid 
!wget https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx
# lub alternatywne źródło json
!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=17ajoD_G-2OTt6Af7U-VoqIMHscFLDn0J' -O covid_prepared.json

import pandas as pd
!pip install xlrd

covid = pd.read_excel("COVID-19-geographic-disbtribution-worldwide.xlsx")
# zmienna covid
covid

# podstawowe statystyki
covid.describe()

# wczytanie pliku covid_prepared.json
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
covid_prepared = pd.read_json("covid_prepared.json")

# zmienna COVID
pd.read_json("covid_prepared.json")

# plt scatter wykres obrazujący przyrost zachorowań w Polsce.
poland_data = covid_prepared[covid_prepared.country == "Poland"]
plt.scatter(poland_data.day_from_begin, poland_data.total_cases)

poland_data = covid_prepared[covid_prepared.country == "Poland"]
plt.scatter(poland_data.day_from_begin, poland_data.deaths_perc)

# poniżej został zaprezentowany przykład w którym wybnieramy wartości total_deaths, deaths..., deaths_movements 
# Warto potestować z innymi atrybutami aby zobaczyć jak zachowa się algorytm budowy klastra
plt.rcParams['figure.dpi'] = 200
pivoted = covid_prepared.pivot(index='country', columns='day_from_begin', values=[
    "total_deaths",
    "deaths",
    "deaths_perc",
    "total_cases",
    "cases",
    "cases_movements",
    "deaths_movements"])

# czesc rekordów ma wartość pustą(null), w takim wypadku należy zamienic wartość pustą(null) na 0
pivoted = pivoted.fillna(0.0)
pivoted.columns = ['{}_{}'.format(*col).strip() for col in pivoted.columns.values]
pivoted

pivoted_norm = pd.DataFrame(pivoted)
# jako parametr method ustaw: ward lub average, centroid ... pelna lista jest w https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# pivoted_norm = normalize(pivoted)
clusters_results = linkage(pivoted_norm, method='complete' )


dendrogram(
    clusters_results,
    labels=pivoted.index,
    leaf_rotation=90.,
    leaf_font_size=4
)
plt.show()

# zmodyfikowany parametr metric na jeden z : ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
clusters_results = linkage(pivoted_norm,method='complete', metric='braycurtis' )
 
dendrogram(
    clusters_results,
    labels=pivoted.index,
    leaf_rotation=90.,
    leaf_font_size=4
)
plt.show()
