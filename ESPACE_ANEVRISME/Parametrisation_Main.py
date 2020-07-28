import numpy as np
import matplotlib.pyplot as plt
import time
import trimesh
from math import *

import Gestion_Fichiers as gf
import BSplines_Utilities as bs
import Parametrisation_Utilities as pru

gf.Reload(bs)
gf.Reload(gf)
gf.Reload(pru)

def Parametrisation(CONTROL, file_stl):

    start_time = time.time()

    ################ INITIALISATION DU MESH ###################

    aorte = trimesh.load_mesh(file_stl)
    #new_vox = Coupe_Voxelisation(CONTROL, aorte)
    #nb_voxel = new_vox.filled_count
    #print("Nombre total de voxel dans la partie considérée : ", nb_voxel)

    ######################## PARAMETRES #######################

    print("------- PARAMETRES -------")
    liste_t = np.linspace(0, 1, 50)
    print("Nombre de coupures : ", len(liste_t))
    liste_theta = np.linspace(0, 2*pi, 150)
    print("Nombre de theta entre 0 et 2pi : ", len(liste_theta))
    degree = 3
    print("Le degré des BSplines est de : ", degree)
    knot = bs.Knotvector(CONTROL, degree)
    coeff_fourier = 5
    print("Ordre de la série de fourier : ", coeff_fourier)
    print("--------------------------")

    ############################################################

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
        THETA_R_EXP, COORD_PLAN = pru.Modelisation_contour(PASSAGE, COORD, TAN, aorte)


        #EFFECTUE LE FITTING DU MODELE D'ORDRE n VIA UNE SERIE DE FOURIER
        fit = pru.Modele_Fourier(THETA_R_EXP, ordre = coeff_fourier)


        #ON CALCUL LES R CORRESPONDANTS AUX THETA GRACE AU MODELE ET ON RESSORT
        #L'ERREUR DU MODELE
        THETA_R_APPROX, err = pru.Theta_R_Approx(fit, THETA_R_EXP, liste_theta)
        print("Coefficient de détermination : ", round(err, 4))


        #ON RECONSTRUIT LES POINTS RECONSTRUIT DANS LA BASE CANONIQUE
        CONTOUR = pru.Reconstruction_contour(COORD_PLAN, THETA_R_APPROX, PASSAGE)


        coordonnees.append(COORD)
        contour.append(CONTOUR)
        erreur.append(err)


        end_ite = time.time()
        print("Temps pour itération {} est de {}".format(i+1, round(end_ite - start_ite, 3)))

    ############### FIN BOUCLE #################################

    COORDONNEES = np.vstack(coordonnees)
    PAROI = np.vstack(contour)

    isfill = new_vox.is_filled(PAROI)
    nb_true = np.count_nonzero(isfill)

    #print("Pourcentage de point dans les voxels initiaux : ", nb_true/nb_voxel)
    #print("Reste : ", 1 - (nb_true/nb_voxel))

    gf.Write_csv("RECONSTRUCTION_PAROI.csv", PAROI, "x, y, z")

    end_time = time.time()
    print("-----------------------------------------------")
    print("TEMPS TOTAL pour {} points est de {}".format(len(liste_t), round(end_time - start_time, 3)))
    print("ERREUR TOTAL (MOYENNE COEFF DETERMINATION) : ", np.mean(erreur))
    print("-----------------------------------------------")

    return COORDONNEES, PAROI

def Coupe_Voxelisation(CONTROL, mesh):

    mesh = trimesh.load_mesh(mesh)
    vox = mesh.voxelized(0.4)
    indices = vox.sparse_indices
    coord = vox.indices_to_points(indices)
    MATRIX = vox.matrix

    degree = 3
    knot = bs.Knotvector(CONTROL, degree)

    C1, T1, N1, B1 = pru.Point_de_reference(0, CONTROL, knot, degree)
    C2, T2, N2, B2 = pru.Point_de_reference(1, CONTROL, knot, degree)

    d1 = -np.sum(C1*T1)
    d2 = -np.sum(C2*T2)

    liste = []
    for i in range(np.shape(coord)[0]):
        val1 = Plan(coord[i,:],T1,d1)
        val2 = Plan(coord[i,:],T2,d2)
        if val1 < 0 or val2 > 0 :
            liste.append(i)

    NEW = coord[liste]
    IND = vox.points_to_indices(NEW)

    for i in range(np.shape(NEW)[0]):
        c1 = IND[i,0]
        c2 = IND[i,1]
        c3 = IND[i,2]
        MATRIX[c1,c2,c3] = False

    return MATRIX

def Plan(COORD, TAN, d):
    return np.sum(COORD*TAN) + d
