import numpy as np
import time
import matplotlib.pyplot as plt
import trimesh
from math import *

from outils import Gestion_Fichiers as gf
from outils import BSplines_Utilities as bs
from espace import Parametrisation_Utilities as pru

gf.Reload(bs)
gf.Reload(pru)

def Parametrisation(CONTROL, aorte, liste_t, liste_theta, degree, coeff_fourier):
    """Permet la reconstruction d'une géométrie. Retourne les coordonnées de la centerline
    ainsi que les contours de la reconstruction."""

    knot = bs.Knotvector(CONTROL, degree)

    coordonnees = []
    contour = []
    erreur = []

    ############### BOUCLE POUR CHAQUE COUPURE #################

    for i in range(len(liste_t)):
        print(" --------- Coupure numéro : {} au point {} ---------- ".format(i+1, liste_t[i]))


        start_ite = time.time()


        #EXTRAIT LES COORDONNEES DU POINT ET LE REPERE DE FRENET
        COORD, TAN, NOR, BI = pru.Point_de_reference(liste_t[i], CONTROL, knot, degree)
        X_t = COORD[:,0]
        Y_t = COORD[:,1]
        Z_t = COORD[:,2]


        #CALCUL DE LA MATRICE DE PASSAGE DE LA BASE CANONIQUE A LA BASE DEFINIE PAR
        #LE REPERE DE FRENET
        PASSAGE = pru.Matrice_de_passage(TAN, NOR, BI)


        #COUPURE DE L'AORTE PAR LE PLAN (N/B). POUR TOUT POINT DE LA COUPURE, RESSORT
        #SON ANGLE / NORMALE (COLONNE 0) ET SA DISTANCE / COORD (COLONNE 2).
        #PERMET D'EXPRIMER UN MODELE VIA UNE SERIE DE FOURIER.
        THETA_R_EXP, COORD_PLAN, COUPURE_PLAN = pru.Modelisation_contour(PASSAGE, COORD, TAN, aorte)


        #EFFECTUE LE FITTING DU MODELE D'ORDRE n VIA UNE SERIE DE FOURIER
        fit = pru.Modele_Fourier(THETA_R_EXP, ordre = coeff_fourier)


        #ON CALCUL LES R CORRESPONDANTS AUX THETA GRACE AU MODELE ET ON RESSORT
        #L'ERREUR DU MODELE
        THETA_R_APPROX, err = pru.Theta_R_Approx(fit, THETA_R_EXP, liste_theta)
        print("Coefficient de détermination : ", round(err, 4))


        #ON RECONSTRUIT LES POINTS RECONSTRUIT DANS LA BASE CANONIQUE
        CONTOUR = pru.Reconstruction_contour(COORD_PLAN, THETA_R_APPROX, PASSAGE, COUPURE_PLAN)


        coordonnees.append(COORD)
        contour.append(CONTOUR)
        erreur.append(err)


        end_ite = time.time()
        print("Temps pour itération {} est de {}".format(i+1, round(end_ite - start_ite, 3)))

    ############### FIN BOUCLE #################################

    COORDONNEES = np.vstack(coordonnees)
    PAROI = np.vstack(contour)

    print("-----------------------------------------------")
    print("ERREUR TOTAL (MOYENNE COEFF DETERMINATION) : ", np.mean(erreur))
    print("-----------------------------------------------")

    return COORDONNEES, PAROI, np.mean(erreur)
