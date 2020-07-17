import numpy as np
from math import *
from random import *
from scipy import interpolate
import matplotlib.pyplot as plt

import BSplines_Utilities as bs
import Statistiques_Utilities as stats
import Gestion_Fichiers as gf
import Base_Centerline_Utilities as base_center
import Base_Contour_Utilities as base_contour

gf.Reload(gf)
gf.Reload(bs)
gf.Reload(base_center)
gf.Reload(base_contour)
gf.Reload(stats)

def Main_base_centerline(liste_control, nb_points, degree):

    #ANALYSE PROCRUSTEENNE DES POINTS DE CONTROLE
    procrust_control, disparity = base_center.Procrustes(liste_control)

    coeff_dilatation = base_center.Dilatation(liste_control)
    print(coeff_dilatation)

    #RECONSTRUCTION DES BSPLINES AVEC BEAUCOUP DE POINTS POUR L'APPROX
    procrust_bsplines = base_center.Construction_bsplines(procrust_control, 200, degree)

    #MATRICE DONT LES COLONNES SONT LES COEFFICIENTS DE L'APPROX POLY
    MAT = base_center.Matrice_coefficients(procrust_bsplines)

    #CREE UNE BASE ORTHONORMALE (REDUITE?) DE nb_vect_base VECTEURS
    BASE_CENTERLINE = base_center.POD(MAT)

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

    #ON EXTRAIT LES COEFFS DES SERIES DE FOURIER POUR 1 t ET POUR TOUTES LES GEOMETRIES
    #ON ORGANISE CES COEFFICIENTS DANS UNE MATRICE POUR 1 t DE TAILLE (nb_coeff X nb_mesh)
    #CHAQUE ELEMENT DE mat_coeff EST UNE MATRICE DE COEFFS DESTINEES A ETRE TRANSFORMEE EN
    #UNE BASE ORTHONORMALE

    MAT, RAYON, determination = base_contour.Matrice_coefficients(liste_control, liste_mesh, liste_t, degree, n_ordre)
    print("La moyenne des coeffs de determination est de : ", np.mean(determination))

    #ON TRANSFORME CHAQUE MATRICE EN UNE BASE (REDUITE?) DE nb_vect_base
    #ORTHONORMALE ET L'ON EFFECTUE LES VERIFICATIONS.
    BASE_CONTOUR = base_center.POD(MAT)

    BASE_RAYON = base_center.POD(RAYON)

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

def Main_interpolation(liste_base, liste_t, target):

    a, b = base_contour.find_interval(liste_t, target)
    print(liste_t[a])
    print(liste_t[b])
    PHI = liste_base[a]
    PSY = liste_base[b]
    val = (target - liste_t[a])/(liste_t[b] - liste_t[a])
    print(val)
    INTERPOL = base_contour.Solution_geodesique(PHI, PSY, val)

    return INTERPOL

def Main_Generation(BASE_CENTERLINE, BASE_CONTOUR, BASE_RAYON, liste_t, t_anevrisme):

    #CREATION BASE RAYON ALEATOIRE
    alea_rayon = np.random.rand(1, np.shape(BASE_RAYON)[1])
    RAY = alea_rayon*BASE_RAYON
    func = interpolate.interp1d(np.asarray(liste_t), RAY[:,0])

    #CREATION DE LA CENTERLINE DE L'ANEVRISME
    alea_centerline = np.random.rand(1, np.shape(BASE_CENTERLINE)[1])
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
        a0 = 1.e-1*func(t_anevrisme[i])
        #VECTEUR CONTOUR ALEATOIRE
        alea_contour = np.random.rand(1, np.shape(BASE_CONTOUR)[1])
        vect = 1.e-1*np.sum(BASE_CONTOUR*alea_contour, axis = 1)
        step2 = int(len(vect[1:])/2)
        coeff_a = vect[1:step2+1]
        coeff_b = vect[step2+1:]
        R = base_contour.Compute_Serie_Fourier(theta, a0, coeff_a, coeff_b)
        TAB = np.hstack((theta[np.newaxis].T, R[np.newaxis].T))
        """
        plt.figure()
        plt.plot(TAB[:,0], TAB[:,1])
        plt.title("R-Theta")
        plt.xlabel("Theta")
        plt.ylabel("R")
        plt.show()
        """
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

    gf.Write_csv("COORD.csv", COORD, "x, y, z, lab")
    gf.Write_csv("CONTOUR.csv", L, "x, y, z, lab")

    gf.Write_csv("ANEVRISME.csv", ANEVRISME, "x, y, z, label")

    return COORD, L, ANEVRISME
