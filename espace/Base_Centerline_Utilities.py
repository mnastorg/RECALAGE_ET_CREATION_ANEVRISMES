###################################################################################################################################
###################################### MODULE DES FONCTIONS POUR LA CREATION DE LA BASE POD CENTERLINE ############################
###################################################################################################################################

import numpy as np
import sys
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from scipy.linalg import svd
from math import *

from outils import Gestion_Fichiers as gf
from outils import BSplines_Utilities as bs

gf.Reload(bs)

def Procrustes(liste_control):
    """ Fonction permettant l'anaylse procrustéenne, i.e permet d'étudier et de faire des
    comparaisons entre différentes Splines représentatives de centerline d'anévrismes """

    REF = liste_control[0]

    result = []
    disparity = []

    for i in  liste_control:
        mtx1, mtx2, disp = procrustes(REF, i)
        result.append(mtx2)
        disparity.append(disp)

    return result, disparity

def Dilatation(liste_control):

    REF = liste_control[0]
    NEW = REF - np.mean(REF, 0)
    scale = np.linalg.norm(NEW)

    return scale

def Construction_bsplines(procrust_control, nb_points, degree):
    """ Construit les BSplines à partir des points de controle qui ont subi
    une analyse procrustéenne."""

    procrust_bsplines = []

    t = np.linspace(0, 1, nb_points)

    for i in range(len(procrust_control)):
        CONTROL = procrust_control[i]
        KNOT = bs.Knotvector(CONTROL, degree)
        BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(CONTROL, KNOT, degree, t, dimension = 3)
        BSPLINES = np.insert(BSPLINES, 3, i, axis = 1)
        procrust_bsplines.append(BSPLINES)

    #gf.Write_csv("PROCRUST_BSPLINES.csv", np.vstack(procrust_bsplines), "x, y, z, num")

    return procrust_bsplines

def Matrice_coefficients(bsplines_procrust):
    """Effectue une approximation polynomiale d'ordre N (donc N+1 coeffs) pour les coords
    X, Y et Z de la Bsplines. On ressort les coefficients de l'approximation et on les place
    dans une matrice. La matrice des coefficients sera donc de taille (3*(N+1), len(bsplines_procrust))."""

    coeff_X = []
    coeff_Y = []
    coeff_Z = []

    t = np.linspace(0, 1, np.shape(bsplines_procrust[0])[0])

    for coord in range(3):

        for i in range(len(bsplines_procrust)) :

            coeffs = np.polyfit(t, bsplines_procrust[i][:,coord], deg = 5)

            if coord == 0 :
                coeff_X.append(coeffs)
            elif coord == 1 :
                coeff_Y.append(coeffs)
            else :
                coeff_Z.append(coeffs)

    liste_concat = []

    for x, y, z in zip(coeff_X, coeff_Y, coeff_Z):
        concat = np.concatenate((x,y,z))
        liste_concat.append(concat)

    MAT = np.vstack(liste_concat).T

    return MAT

def Creation_base(MAT, nb_vect_base):
    """ Permet la création d'une base orthonormale avec un nombre
    nb_vect_base de vecteur dans la base"""

    return Orthonormalisation(MAT, nb_vect_base)

def POD(MAT, string):
    """ Effectue la proper orthogonal decomposition sur le critère de Kaiser"""

    taille = np.shape(MAT)

    PROJ = np.dot(MAT, MAT.T)
    val, VECT = np.linalg.eigh(PROJ)
    moy = np.mean(val)
    indices = (np.where(val >= moy))[0]    
    BASE = VECT[:,indices]

    s = np.sum(val)
    plt.plot(np.arange(len(val)), (val[::-1]/s)*100, '-o')
    plt.title("Pourcentage part des valeurs propres pour " + string)
    plt.xlabel("Valeur propre")
    plt.ylabel("Pourcentage variance")
    plt.show()

    return BASE

def Gram_schmidt(X):
    Q, R = np.linalg.qr(X, mode = 'reduced')
    return Q

def Proj(v,u):

    proj = np.dot(v,u)*u

    return proj

def Orthonormalisation(MATRIX, nbr_vect):

    N = np.shape(MATRIX)[0]
    k = np.shape(MATRIX)[1]

    ortho = []
    norm = []
    indice = []

    x_gamma = MATRIX[:,0]
    e = x_gamma/np.linalg.norm(x_gamma)

    ortho.append(e)

    for i in range(1,k):
        v = MATRIX[:,i]
        sum = np.zeros(N)

        for i in range(len(ortho)):
            sum += Proj(v,ortho[i])


        u = v - sum
        norm.append(np.linalg.norm(u))
        e = u/np.linalg.norm(u)
        ortho.append(e)

    if nbr_vect > k:
        print("ERREUR, veuillez saisir un nombre moins important de vecteurs de base à conserver")
        sys.exit()

    else:
        m = 0
        while m < nbr_vect - 1:
            indice.append(norm.index(min(norm)))
            norm[norm.index(min(norm))] = 100*max(norm)
            m = m + 1

    ORTHO = np.zeros((N, nbr_vect))
    ORTHO[:, 0] = ortho[0]
    for i in range(len(indice)):
        ORTHO[:, i+1] = ortho[indice[i] + 1]

    return ORTHO

def Verification_base(BASE):
    """Retourne la différence entre la matrice identité et t_BASE*BASE. On s'attend
    à ce que cette différence soit minime."""

    I = np.eye(np.shape(BASE)[1])
    VERIF = np.dot(BASE.T, BASE)

    return np.linalg.norm(I - VERIF)

def Verification_Erreur_Moyenne(MAT, BASE):
    """On regarde tous les vecteurs qui composent la base initiale
    MAT. Pour chaque vecteur on regarde sa norme entre lui-meme et son
    projeté dans la nouvelle base (BASE*t_BASE)."""

    l = []
    PROJ = np.dot(BASE, BASE.T)
    for i in range(np.shape(MAT)[1]) :
        VECT = MAT[:,i]
        l.append(np.linalg.norm(VECT - np.dot(PROJ, VECT)))

    return l
