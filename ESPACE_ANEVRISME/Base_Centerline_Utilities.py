import numpy as np
from scipy.spatial import procrustes
from math import *

import Gestion_Fichiers as gf
import BSplines_Utilities as bs

gf.Reload(gf)
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
    if nb_vect_base == np.shape(MAT)[1]:
        return Gram_schmidt(MAT)

    else :
        return 0

def Gram_schmidt(X):
    Q, R = np.linalg.qr(X, mode = 'reduced')
    return Q

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
    for i in range(np.shape(MAT)[1]) :
        VECT = MAT[:,i]
        PROJ = np.dot(BASE, BASE.T)
        l.append(np.linalg.norm(VECT - np.dot(PROJ, VECT)))

    return l