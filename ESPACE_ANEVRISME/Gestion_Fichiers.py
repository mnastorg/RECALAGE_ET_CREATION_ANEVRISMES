import numpy as np
import pandas as pd
import importlib

def Lecture_csv_nurea(file_nurea):
    """Retourne un tableau qui lit le squelette de Nurea avec les paramètre x, y, z, label, isSail, isLeaf"""
    #LECTURE DU FICHIER CSV
    DF = pd.read_csv(file_nurea)
    #LECTURE DES PARAMETRES QUI NOUS INTERESSENT
    x = np.asarray(DF[' x'])
    y = np.asarray(DF[' y'])
    z = np.asarray(DF[' z'])
    lab = np.asarray(DF[' Label'])
    is_sail = np.asarray(DF[' isBifurcation'])
    is_leaf = np.asarray(DF[' isLeaf'])
    #CREATION DU TABLEAU RELATIF AU CSV
    if (len(x) == len(y) and len(x)==len(z)):
        NUREA = np.zeros((len(x),6))
        NUREA[:,0] = x
        NUREA[:,1] = y
        NUREA[:,2] = z
        NUREA[:,3] = lab
        NUREA[:,4] = is_sail
        NUREA[:,5] = is_leaf
    else :
        print("ERREUR PAS LE MEME NBR DE COORDONNEES")

    return NUREA

def Lecture_csv_pcl(file_csv_pcl):
    """Retourne un tableau qui lit le squelette de Nurea avec les paramètre x, y, z, label, isSail, isLeaf"""
    #LECTURE DU FICHIER CSV
    DF = pd.read_csv(file_csv_pcl)
    #LECTURE DES PARAMETRES QUI NOUS INTERESSENT
    x = np.asarray(DF['# x'])
    y = np.asarray(DF[' y'])
    z = np.asarray(DF[' z'])
    #CREATION DU TABLEAU RELATIF AU CSV
    if (len(x) == len(y) and len(x)==len(z)):
        PCL = np.zeros((len(x),3))
        PCL[:,0] = x
        PCL[:,1] = y
        PCL[:,2] = z
    else :
        print("ERREUR PAS LE MEME NBR DE COORDONNEES")

    return PCL

def Lecture_csv_levelset(file_csv_levelset):
    """Retourne un tableau qui lit le squelette de Nurea avec les paramètre x, y, z, label, isSail, isLeaf"""
    #LECTURE DU FICHIER CSV
    DF = pd.read_csv(file_csv_levelset)
    #LECTURE DES PARAMETRES QUI NOUS INTERESSENT
    x = np.asarray(DF['# x'])
    y = np.asarray(DF[' y'])
    z = np.asarray(DF[' z'])
    scalar = np.asarray(DF[' scalaire'])

    #CREATION DU TABLEAU RELATIF AU CSV
    if (len(x) == len(y) and len(x)==len(z)):
        LEVELSET = np.zeros((len(x),4))
        LEVELSET[:,0] = x
        LEVELSET[:,1] = y
        LEVELSET[:,2] = z
        LEVELSET[:,3] = scalar

    else :
        print("ERREUR PAS LE MEME NBR DE COORDONNEES")

    return LEVELSET

def Write_csv(NOM, CONCAT, HEAD):
    """Ecrit un fichier csv 'NOM' issu de CONCAT avec un header"""
    np.savetxt(NOM, CONCAT, delimiter = ", ", header = HEAD)

def Reload(module):
    importlib.reload(module)
