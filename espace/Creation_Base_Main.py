###################################################################################################################################
###################################### MODULE POUR CREATION DES BASES POD POUR UN ENSEMBLE DE GEOMETRIES ##########################
###################################################################################################################################

import numpy as np
from math import *
#from random import *
from scipy import interpolate
import matplotlib.pyplot as plt

from outils import Gestion_Fichiers as gf
from outils import BSplines_Utilities as bs
from outils import Statistiques_Utilities as stats
from espace import Base_Centerline_Utilities as base_center
from espace import Base_Contour_Utilities as base_contour

gf.Reload(bs)
gf.Reload(base_center)
gf.Reload(base_contour)
gf.Reload(stats)

def Main_base_centerline(liste_control, nb_points, degree):
    """Retourne la base POD des centerlines."""
    #ANALYSE PROCRUSTEENNE DES POINTS DE CONTROLE
    procrust_control, disparity = base_center.Procrustes(liste_control)

    #RECONSTRUCTION DES BSPLINES AVEC BEAUCOUP DE POINTS POUR L'APPROX
    procrust_bsplines = base_center.Construction_bsplines(procrust_control, 200, degree)

    #MATRICE DONT LES COLONNES SONT LES COEFFICIENTS DE L'APPROX POLY
    MAT = base_center.Matrice_coefficients(procrust_bsplines)

    #CREE UNE BASE ORTHONORMALE (REDUITE?) DE nb_vect_base VECTEURS
    BASE_CENTERLINE = base_center.POD(MAT, " base centerline")

    #VERIFIE SI LA BASE EST ORTHONORMALE i.e t_B*B = I
    indice = base_center.Verification_base(BASE_CENTERLINE)
    if indice < 1.e-10 :
        print("La base centerline est orthonormée avec une erreur : ", indice)
    else :
        print("La base centerline n'est pas orthonormée, erreur de : ", indice)

    #ON VERIFIE QUE LA DIFFERENCE MOYENNE ENTRE TOUS LES VECTEURS DE LA BASE
    #ET LEUR PROJETE DANS LA BASE ORTHONORMALE EST INFIME
    l = base_center.Verification_Erreur_Moyenne(MAT, BASE_CENTERLINE)
    print("L'erreur moyenne entre les vecteurs de la base centerline et leur projeté est de : ", np.mean(l))

    return BASE_CENTERLINE

def Main_base_contour(liste_control, liste_mesh, liste_t, degree, n_ordre):
    """Retourne la base POD des contours"""
    #ON EXTRAIT LES COEFFS DES SERIES DE FOURIER POUR 1 t ET POUR TOUTES LES GEOMETRIES
    #ON ORGANISE CES COEFFICIENTS DANS UNE MATRICE POUR 1 t DE TAILLE (nb_coeff X nb_mesh)
    #CHAQUE ELEMENT DE mat_coeff EST UNE MATRICE DE COEFFS DESTINEES A ETRE TRANSFORMEE EN
    #UNE BASE ORTHONORMALE

    MAT, RAYON, determination = base_contour.Matrice_coefficients(liste_control, liste_mesh, liste_t, degree, n_ordre)
    print("La moyenne des coeffs de determination est de : ", np.mean(determination))

    #ON TRANSFORME CHAQUE MATRICE EN UNE BASE (REDUITE?) DE nb_vect_base
    #ORTHONORMALE ET L'ON EFFECTUE LES VERIFICATIONS.
    BASE_CONTOUR = base_center.POD(MAT, " base contour")

    BASE_RAYON = base_center.POD(RAYON, " base rayon")

    indice = base_center.Verification_base(BASE_CONTOUR)
    if indice < 1.e-10 :
        print("La base contour est orthonormée avec une erreur : ", indice)
    else :
        print("La base contour n'est pas orthonormée, erreur de : ", indice)

    l = base_center.Verification_Erreur_Moyenne(MAT, BASE_CONTOUR)
    print("L'erreur moyenne entre les vecteurs de la base contour et leur projeté est de : ", np.mean(l))

    indice2 = base_center.Verification_base(BASE_RAYON)
    if indice2 < 1.e-10 :
        print("La base rayon est orthonormée avec une erreur : ", indice2)
    else :
        print("La base rayon n'est pas orthonormée, erreur de : ", indice2)

    return BASE_CONTOUR, BASE_RAYON

def Main_Generation(BASE_CENTERLINE, BASE_CONTOUR, BASE_RAYON, coeff_dilatation, t_anevrisme):

    """Retourne un anévrisme généré par les bases calculées précédemment."""

    #CREATION BASE RAYON ALEATOIRE
    alea_rayon = 2*np.random.rand(1,np.shape(BASE_RAYON)[1])

    print("L'aléatoire du rayon est de : ", alea_rayon)
    RAY = np.sum(BASE_RAYON*alea_rayon, axis = 1)
    liste_t = np.linspace(0,1,np.shape(RAY)[0])
    func = interpolate.interp1d(np.asarray(liste_t), RAY)

    #CREATION DE LA CENTERLINE DE L'ANEVRISME
    #alea_centerline = 0.2 + 0.8*np.random.rand(1, np.shape(BASE_CENTERLINE)[1])
    alea_centerline = 4*np.random.rand(1, np.shape(BASE_CENTERLINE)[1]) - 2
    alea_centerline[:,1] = np.random.rand(1, 1)

    print("L'aléatoire de la centerline est de : ", alea_centerline)

    coeffs = np.sum(BASE_CENTERLINE*alea_centerline, axis = 1)

    step = int(len(coeffs)/3)
    coeffs_x = coeffs[0:step]
    coeffs_y = coeffs[step:2*step]
    coeffs_z = coeffs[2*step:]
    px = np.poly1d(coeffs_x)
    py = np.poly1d(coeffs_y)
    pz = np.poly1d(coeffs_z)
    der1x = np.polyder(px, m = 1)
    der1y = np.polyder(py, m = 1)
    der1z = np.polyder(pz, m = 1)
    der2x = np.polyder(px, m = 2)
    der2y = np.polyder(py, m = 2)
    der2z = np.polyder(pz, m = 2)

    COORD = np.zeros((len(t_anevrisme),3))
    COORD[:,0] = px(t_anevrisme)
    COORD[:,1] = py(t_anevrisme)
    COORD[:,2] = pz(t_anevrisme)
    COORD = (coeff_dilatation*(COORD))#/np.linalg.norm(COORD, axis = 0)

    DER1 = np.zeros((len(t_anevrisme),3))
    DER1[:,0] = der1x(t_anevrisme)
    DER1[:,1] = der1y(t_anevrisme)
    DER1[:,2] = der1z(t_anevrisme)

    DER2 = np.zeros((len(t_anevrisme),3))
    DER2[:,0] = der2x(t_anevrisme)
    DER2[:,1] = der2y(t_anevrisme)
    DER2[:,2] = der2z(t_anevrisme)

    TAN = stats.tangente(DER1)
    BI = stats.binormale(DER1, DER2)
    NOR = stats.normale(BI, TAN)

    #CREATION DES COUPURES
    liste_contour = []
    theta = np.linspace(0, 2*pi, 150)

    for i in range(len(t_anevrisme)) :
        #CALCUL DU A0
        c = 10*func(t_anevrisme[i])

        #VECTEUR CONTOUR ALEATOIRE
        alea_contour = 0.8 + 0.4*np.random.rand(1,1)
        vect = np.sum(BASE_CONTOUR[:,0]*alea_contour, axis = 1)
        step2 = int(len(vect[1:])/2)
        a0 = c
        coeff_a = vect[1:step2+1]
        coeff_b = vect[step2+1:]

        R = base_contour.Compute_Serie_Fourier(theta, a0, coeff_a, coeff_b)
        TAB = np.hstack((theta[np.newaxis].T, R[np.newaxis].T))
        #COORD
        C = COORD[i,:]
        T = TAN[i,:]
        N = NOR[i,:]
        B = BI[i,:]
        PASSAGE = base_contour.Matrice_de_passage(T, N, B)
        COORD_PLAN = ((np.dot(PASSAGE.T, C.T)).T)
        CONTOUR = base_contour.Reconstruction_contour(COORD_PLAN, TAB, PASSAGE)
        liste_contour.append(CONTOUR)

    COORD = np.insert(COORD, 3, 0, axis = 1)
    L = np.vstack(liste_contour)
    L = np.insert(L, 3, 100, axis = 1)
    ANEVRISME = np.vstack((COORD, L))

    return ANEVRISME
