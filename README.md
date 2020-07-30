# Présentation du travail


## Introduction 


## Installation

Pour faire fonctionner les codes, nous vous proposons la méthode suivante. Nous utilisons le langage de programmation Python et la distribution Anaconda.

Le fichier "stage_anevrismes.yml" est un environnement conda qui permet de se placer dans un cadre offrant toutes les dépendances nécessaires au bon fonctionnement des fichiers. Pour l'installer vous pouvez : 

  - Soit l'installer directement  via le navigateur Anaconda (Environnements ---> Importer)
  - Soit entrer la ligne de commande (Terminal pour Linux/MacOs ou Anaconda-Prompt pour Windows) : **conda env create -f stage_anevrismes.yml**

Vous pouvez vous placer dans l'environnement en tapant : **conda activate stage_anevrismes**.

Pour plus d'informations sur les environnements conda vous pouvez vous référer à la page suivante :  
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Finalement, il suffit de charger le fichier MAIN.py. Par exemple, vous pouvez utiliser Spyder ou bien encore Ipython en tapant : **run MAIN.py**

## Si l'installation ne fonctionne pas 

Des problèmes peuvent apparaître lors de l'importation de l'environnement. Une cause majeure concerne la dépendance entre les systèmes d'exploitation. Une solution peut être de supprimer les packages manuellement dans le fichier "stage_anevrismes.yml" et de les réinstaller par la suite. 

Sinon, vous pouvez installer manuellement les dépendances suivantes (en cherchant sur internet en fonction de vos besoins) qui sont celles requises pour le fonctionnement des fichiers : numpy / trimesh / scikit-learn / scikit-fmm / geomdl / symfit / time / pandas / ipython.


## Organisation du fichier MAIN.py

Le fichier principal MAIN.py regroupe les 3 fonctions suivantes : 
  
  1. Recalage 
  2. Espace_Anevrismes
  3. Parametrisation 
 
Une fois le fichier MAIN.py chargé, il suffit de lancer une des trois fonctions ci-dessus en entrant les bons paramètres. 
 
Les résultats sortent sous format .csv dans les sous-dossiers du dossier "résultats" : fitting, extraction, base, classe_anevrismes et parametrisatoin.

Les fichiers tests ne sont pas disponibles sur cette page car ils sont trop volumineux pour être importés. Cependant, nous pouvons vous les fournir si nécessaire. 

### Fonction Recalage 

Cette fonction permet d'effectuer le recalage d'un squelette initial dans sa géomértie initiale. Elle s'appuie sur l'extraction d'un nuage de point représentant grossièrement (à partir d'un seuil) les points de discontinuité de la levelset de la géométrie. Elle effectue ensuite une recalage optimal au sens des moindres carrés du squelette initial dans ce nuage de point.

Elle prend en paramètres : 
  
  1. Le fichier .stl (maillage de surface) de la géométrie initiale 
  2. Le fichier .csv (squelette initial)
  3. Le fichier .csv du nuage de point initial (défaut 0 --> calcul du nuage)
  4. Une liste_extraction qui permet d'extraire les labels qui nous intéressent si besoin (défaut 0 --> pas d'extraction)

Exemple d'utilisation : 
**Recalage("test_files/stl/p001.stl", "test_files/csv/p001.csv", liste_p001)**

### Fonction Espace_Anevrismes

Cette fonction permet de créer un espace d'anévrismes. On se donne plusieurs géométries et centerlines associées et, via une méthode POD, on créé trois bases réduites : une base pour la centerline, une base pour la paroi et une base pour la distribution des rayons. Ces dernières nous permettent de décrire une classe d'anévrismes et de en générer autant que nous voulons. 

Elle prend en paramètres : 

1. doss_centerline (dossier menant aux .csv des centerlines (décrit par leurs points de contrôles)
2. doss_mesh (dossier menant aux .stl des géométries)
3. generation (un entier qui indique le nombre d'anévrismes aléatoires à générer. Si 0 --> pas d'anévrismes)

Exemple d'utilisation : 
**Espace_Anevrismes("test_files/csv/", "test_files/stl/", generation = 10)**

### Fonction Parametrisation

Cette fonction permet de paramétriser une seule géométrie. Par exemple, si on se donne les points de contrôles d'une BSplines représentant une centerline et la géométrie associée, on s'intéresse à ressortir la paramétrisation associée.

Elle prend en paramètres : 
  1. file_control (le fichier .csv des points de controle)
  2. file_stl (le fichier .stl de la géométrie associée)

Exemple d'utilisation : 
**Parametrisation("test_files/csv_coupure/p001.csv", "test_files/stl/p001.stl")**


