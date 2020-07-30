# Présentation du travail


## Introduction 


## Installation des codes

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


## Fonctionnement des codes

La fichier principal MAIN.py regroupe les 3 fonctions suivantes : 
  
  1. Recalage 
  2. Espace_Anevrisme
  3. Parametrisation 
 
 La description des fonctions utilisées est décrite plus précisément en Annexe du mémoire (.pdf) joint avec ce GitHub.
