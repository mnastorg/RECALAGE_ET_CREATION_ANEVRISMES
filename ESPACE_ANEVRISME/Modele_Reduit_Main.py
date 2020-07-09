import numpy as np
import matplotlib.pyplot as plt
import time
from math import *

import Gestion_Fichiers as gf
import Modele_Reduit_Utilities as mru
import BSplines_Utilities as bs

gf.Reload(bs)
gf.Reload(gf)
gf.Reload(mru)

def Main_Modele_Reduit(liste_t, liste_theta, CONTROL, mesh):

    start_time = time.time()

    parametrisation_totale = []
    new_points = []
    cercle = []
    erreur = []

    degree = 3
    knot = bs.Knotvector(CONTROL, degree)

    for i in range(len(liste_t)):
        print(" --------- Itération numéro : {} ---------- ".format(i+1))
        start_ite = time.time()


        #EXTRAIT LES COORDONNEES DU POINT ET LE REPERE DE FRENET
        COORD, TAN, NOR, BI = mru.Point_de_reference(liste_t[i], CONTROL, knot, degree)
        X_t = COORD[:,0]
        Y_t = COORD[:,1]
        Z_t = COORD[:,2]


        #CALCUL DE LA MATRICE DE PASSAGE DE LA BASE CANONIQUE A LA BASE DEFINIE PAR
        #LE REPERE DE FRENET
        PASSAGE = mru.Matrice_de_passage(TAN, NOR, BI)

        #COUPURE DE L'AORTE PAR LE PLAN (N/B). POUR TOUT POINT DE LA COUPURE, RESSORT
        #SON ANGLE / NORMALE (COLONNE 0) ET SA DISTANCE / COORD (COLONNE 2).
        #PERMET D'EXPRIMER UN MODELE VIA UNE SERIE DE FOURIER.
        THETA_R_EXP, COORD_PLAN, CERCLE = mru.Modelisation_contour(PASSAGE, COORD, TAN, mesh)
        cercle.append(CERCLE)


        #EFFECTUE LE FITTING DU MODELE D'ORDRE n VIA UNE SERIE DE FOURIER
        fit = mru.Modele_Fourier(THETA_R_EXP, ordre = 5)


        #ON CALCUL LES R CORRESPONDANTS AUX THETA GRACE AU MODELE ET ON RESSORT
        #L'ERREUR DU MODELE
        THETA_R_APPROX, err = mru.Theta_R_Approx(fit, THETA_R_EXP, liste_theta)
        print("L'erreur est de : ", round(err, 4))
        erreur.append(err)
        parametrisation_totale.append([X_t, Y_t, Z_t, THETA_R_APPROX])


        #ON RECONSTRUIT LES POINTS RECONSTRUIT DANS LA BASE CANONIQUE
        NEW_POINTS = mru.Reconstruction_contour(COORD_PLAN, THETA_R_APPROX, PASSAGE)
        new_points.append(NEW_POINTS)

        """
        plt.figure()
        plt.scatter(COUPURE_PLAN[:,0], COUPURE_PLAN[:,1])
        plt.scatter(NEW_POINTS[:,0], NEW_POINTS[:,1])
        plt.scatter(COORD_PLAN[:,0], COORD_PLAN[:,1], color = 'green')
        plt.quiver(COORD_PLAN[:,0], COORD_PLAN[:,1], N_PLAN[:,0], N_PLAN[:,1], color = "red", label = "normale")
        plt.quiver(COORD_PLAN[:,0], COORD_PLAN[:,1], B_PLAN[:,0], B_PLAN[:,1], color = "blue", label = "binormale")
        plt.legend()
        plt.title("Coupure de l'aorte avec le nouveau repère")
        plt.show()
        """

        end_ite = time.time()
        print("Temps pour itération {} est de {}".format(i+1, round(end_ite - start_ite, 3)))

    end_time = time.time()
    print("Temps total pour {} points est de {}".format(len(liste_t), round(end_time - start_time, 3)))
    print("L'erreur totale est de : ", np.mean(erreur))

    return parametrisation_totale, new_points, cercle
