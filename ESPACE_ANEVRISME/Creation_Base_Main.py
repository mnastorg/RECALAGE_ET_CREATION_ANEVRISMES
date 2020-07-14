import numpy as np
from math import *

import BSplines_Utilities as bs
import Gestion_Fichiers as gf
import Base_Centerline_Utilities as base_center
import Base_Contour_Utilities as base_contour

gf.Reload(gf)
gf.Reload(bs)
gf.Reload(base_center)
gf.Reload(base_contour)

def Main_base_centerline(liste_control, nb_points, degree, nb_vect_base):

    #ANALYSE PROCRUSTEENNE DES POINTS DE CONTROLE
    procrust_control, disparity = base_center.Procrustes(liste_control)

    #RECONSTRUCTION DES BSPLINES AVEC BEAUCOUP DE POINTS POUR L'APPROX
    procrust_bsplines = base_center.Construction_bsplines(procrust_control, 200, degree)

    #MATRICE DONT LES COLONNES SONT LES COEFFICIENTS DE L'APPROX POLY
    MAT = base_center.Matrice_coefficients(procrust_bsplines)

    #CREE UNE BASE ORTHONORMALE (REDUITE?) DE nb_vect_base VECTEURS
    BASE = base_center.Creation_base(MAT, nb_vect_base)

    #VERIFIE SI LA BASE EST ORTHONORMALE i.e t_B*B = I
    indice = base_center.Verification_base(BASE)
    if indice < 1.e-14 :
        print("La base centerline est orthonormée avec une erreur : ", indice)
    else :
        print("La base centerline n'est pas orthonormée, erreur de : ", indice)

    #ON VERIFIE QUE LA DIFFERENCE MOYENNE ENTRE TOUS LES VECTEURS DE LA BASE
    #ET LEUR PROJETE DANS LA BASE ORTHONORMALE EST INFIME
    l = base_center.Verification_Erreur_Moyenne(MAT, BASE)
    print("L'erreur moyenne entre les vecteurs de la base centerline et leur projeté est de : ", np.mean(l))

    return BASE

def Main_base_contour(liste_control, liste_mesh, liste_t, nb_vect_base, degree, n_ordre):

    #ON EXTRAIT LES COEFFS DES SERIES DE FOURIER POUR 1 t ET POUR TOUTES LES GEOMETRIES
    #ON ORGANISE CES COEFFICIENTS DANS UNE MATRICE POUR 1 t DE TAILLE (nb_coeff X nb_mesh)
    #CHAQUE ELEMENT DE mat_coeff EST UNE MATRICE DE COEFFS DESTINEES A ETRE TRANSFORMEE EN
    #UNE BASE ORTHONORMALE
    mat_coeff = []
    erreur = []
    for i in liste_t :
        MAT, determination = base_contour.Matrice_coefficients(liste_control, liste_mesh, i, degree, n_ordre)
        mat_coeff.append(MAT)
        erreur.append(determination)

    print("La moyenne des coeffs de determination est de : ", np.mean(erreur))

    #ON TRANSFORME CHAQUE MATRICE EN UNE BASE (REDUITE?) DE nb_vect_base
    #ORTHONORMALE ET L'ON EFFECTUE LES VERIFICATIONS.
    base = []
    for i in mat_coeff:
        BASE = base_center.Creation_base(i, nb_vect_base)
        base.append(BASE)
        indice = base_center.Verification_base(BASE)
        if indice < 1.e-14 :
            print("La base contour est orthonormée avec une erreur : ", indice)
        else :
            print("La base contour n'est pas orthonormée, erreur de : ", indice)
        l = base_center.Verification_Erreur_Moyenne(i, BASE)
        print("L'erreur moyenne entre les vecteurs de la base contour et leur projeté est de : ", np.mean(l))

    return base
