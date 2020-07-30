import numpy as np
import trimesh
import time
import os

import outils
import recalage
import espace

from outils import Gestion_Fichiers as gf

from recalage import *
from outils import *
from espace import *

#SI ON VEUT RELOAD UN MODULE FAIRE COMMANDE gf.Reload(nom_dossier.nom_module)
#EXEMPLE ! : (gf.Reload(recalage.Nuage_Level_Set_Main))

#LISTE D'EXTRACTION DES LABELS
p001 = [[19,25], 'p001']
p002 = [[6,8], 'p002']
p004 = [[9], 'p004']
p008 = [[6,8], 'p008']
p011 = [[7], 'p011']
p013 = [[6], 'p013']
p015 = [[7], 'p015']

def Recalage(file_stl, file_csv, file_pcl = 0, liste_extraction = 0):

    """A partir d'un squelette fourni par NUREA (.csv) ainsi qu'une géométrie (.stl) maillage de surface,
    effectue le recalage du squelette dans la géométrie. Si nuage de point déjà calculé, entrer le chemin dans
    file_pcl (defaut 0 ---> calcul le nuage) et si on souhaite extraire des labels en particulier, ajouter la
    liste (identique à ci-dessus) des labels dans liste_extraction (defaut 0 -----> pas d'extraction)."""

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ DEBUT DU RECALAGE --------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    start_time = time.time()

    NUREA = gf.Lecture_csv_nurea(file_csv)

    ############################################################################
    ######################## PARAMETRES UTILISES ###############################
    print("-------------------------- PARAMETRES -----------------------------")

    #VOXELISATION
    print("VOXELISATION :")
    taille_voxel = 0.8
    print("Taille_voxel : ", taille_voxel)
    seuil_laplacien = 0.7
    print("Seuil_laplacien : {} \n".format(seuil_laplacien))

    #RECALAGE
    print("RECALAGE : ")
    degree_bsplines = 2
    print("Degrée des BSplines : ", degree_bsplines)
    iter = 5
    print("Maximum d'itérations : ", iter)
    tolerance = 1.e-4
    print("Tolérance optimisation : ", tolerance)
    rigidite = 1.e-6
    print("Rigidité : {} \n".format(rigidite))

    #EXTRACTION SI BESOIN
    print("EXTRACTION : ")
    degree_extract = 3
    print("Degré pour extraction : ", degree_extract)
    nb_control_extract = 5
    print("Nombre pts de controle extraction : ", nb_control_extract)
    iter_extract = 50
    print("Maximum itérations pour extraction : ", iter_extract)
    tol_extract =  1.e-5
    print("Tolérance extraction : ", tol_extract)
    rig_extract = 0
    print("Rigidité extraction : ", rig_extract)
    print("-------------------------------------------------------------------")
    ############################################################################
    ############################################################################

    if file_pcl == 0 :

        print('---------------------------------------------------------------------------------------------------------------------------')
        print('--------------------------------------- CREATION DU NUAGE DE POINTS PAR LEVEL SET -----------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')

        PCL, LEVELSET = recalage.Nuage_Level_Set_Main.Main_nuage_level_set(file_stl, taille_voxel, seuil_laplacien)
        gf.Write_csv("resultats/fitting/PCL.csv", PCL, "x, y, z")
        gf.Write_csv("resultats/fitting/LEVELSET.csv", LEVELSET, "x, y, z, scalar")

        print('---------------------------------------------------------------------------------------------------------------------------')
        print('--------------------------------------- FIN CREATION DU NUAGE DE POINTS PAR LEVEL SET -------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')

    else :

        print('---------------------------------------------------------------------------------------------------------------------------')
        print('--------------------------------------- LECTURE DU FICHIER CSV PCL --------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')

        PCL = gf.Lecture_csv_pcl(file_pcl)

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------ DEBUT DE L INITIALISATION ------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    RECONSTRUCTION, liste_controle, liste_bsplines, liste_knot, liste_basis_der2, liste_t, merge = recalage.Initialisation_Main.Main_Initialisation(NUREA, degree_bsplines)

    gf.Write_csv("resultats/fitting/INIT_RECONSTRUCTION.csv", RECONSTRUCTION, "x, y, z, label, is_sail, is_leaf, TX, TY, TZ, NX, NY, NZ, BX, BY, BZ, courbure, longueur")
    gf.Write_csv("resultats/fitting/INIT_CONTROLES.csv", np.vstack(liste_controle), "x, y, z, label, is_sail, is_leaf")
    gf.Write_csv("resultats/fitting/INIT_BSPLINES.csv", np.vstack(liste_bsplines), "x, y, z, label, is_sail, is_leaf, TX, TY, TZ, NX, NY, NZ, BX, BY, BZ")
    gf.Write_csv("resultats/fitting/INIT_SAILLANTS.csv", ((np.vstack(liste_bsplines))[(np.vstack(liste_bsplines))[:,4] == 1])[:,0:3], "x, y, z")

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('-------------------------------------------- FIN DE L INITIALISATION ------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------- DEBUT DU FITTING ----------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    liste_reconstruction = [liste_controle, liste_bsplines, liste_knot, liste_basis_der2, liste_t, merge]
    new_control, new_bsplines, new_der1, new_der2 = recalage.Fitting_Complexe_Main.Main_fitting_complexe(PCL, liste_reconstruction, degree = degree_bsplines, max_iter = iter, tol = tolerance, rig = rigidite)

    gf.Write_csv("resultats/fitting/FIT_CONTROL.csv", np.vstack(new_control), "x, y, z, label, is_sail, is_leaf")
    gf.Write_csv("resultats/fitting/FIT_BSPLINES.csv", np.vstack(new_bsplines), "x, y, z, label, is_sail, is_leaf")
    gf.Write_csv("resultats/fitting/FIT_SAILL.csv", ((np.vstack(new_bsplines))[(np.vstack(new_bsplines))[:,4] == 1])[:,0:3], "x, y, z")

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ FIN DU FITTING -----------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    if liste_extraction != 0 :

        print('---------------------------------------------------------------------------------------------------------------------------')
        print('-------------------------------------- DEBUT EXTRACTION ET SMOOTHING DE LA COURBE PRINCIPALE ------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')

        PRINCIPALE, centerline_parametres = recalage.Extraction_Centerline_Main.Main_Extraction_Centerline_Labels(new_bsplines, liste_extraction[0], nb_control = nb_control_extract, degree = degree_extract)

        gf.Write_csv("resultats/extraction/EXTRACT_PRINCIPALE.csv", PRINCIPALE, "x, y, z")
        gf.Write_csv("resultats/extraction/EXTRACT_CONTROL.csv", centerline_parametres[0], "x, y, z")
        gf.Write_csv("resultats/extraction/EXTRACT_BSPLINES.csv", centerline_parametres[1], "x, y, z")

        #ON EFFECTUE LE FITTING DE LA BSPLINES APPROXIMEE SUR LA BSPLINES PRINCIPALE POUR LA RENDRE LISSE
        liste_smoothing = [centerline_parametres[0], centerline_parametres[1], centerline_parametres[2], centerline_parametres[5], centerline_parametres[6]]
        SMOOTH_CONTROL, SMOOTH_BSPLINES = recalage.Fitting_Simple_Main.Main_fitting_simple(PRINCIPALE, liste_smoothing, degree = degree_extract, max_iter = iter_extract, tol = tol_extract, rig = rig_extract)

        gf.Write_csv("resultats/extraction/SMOOTH_CONTROL.csv".format(liste_extraction[1]), SMOOTH_CONTROL, "x, y, z")
        gf.Write_csv("resultats/extraction/SMOOTH_BSPLINES.csv", SMOOTH_BSPLINES, "x, y, z, TX, TY, TZ, NX, NY, NZ, BX, BY, BZ, courbure")

        print('---------------------------------------------------------------------------------------------------------------------------')
        print('---------------------------------------- FIN EXTRACTION ET SMOOTHING DE LA COURBE PRINCIPALE ------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')

        end_time = time.time()
        print("Temps total de la méthode : ", round(end_time - start_time, 2))

        return 1

    return 0

def Espace_Anevrisme(doss_centerline, doss_mesh, generation = 0):
    """Fonction permettant de créer les bases réduites POD à partir d'un certain nombre de centerline
    et géométries associées. Si génération différent 0 alors cela créé autant d'anévrismes aléatoire que
    génération."""

    start_time = time.time()

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ DEBUT ESPACE ANEVRISMES --------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    ############################################################################
    ######################## PARAMETRES UTILISES ###############################
    print("-------------------------- PARAMETRES -----------------------------")
    #BASE CENTERLINE
    print("BASE CENTERLINE : ")
    nb_points_reconstruction = 200
    print("Nombre de points de reconstruction : ", nb_points_reconstruction)
    degree_reconstruction = 3
    print("Degré reconstruction : {} . Attention il doit être égale à celui du recalage \n".format(degree_reconstruction))

    #BASE CONTOUR
    print("BASE CONTOUR : ")
    nb_coupure = 10
    print("Nombre de coupure par géométrie : ", nb_coupure)
    liste_t = np.linspace(0, 1, nb_coupure)
    theta = np.linspace(0, 2*np.pi, 150)
    print("Nombre de theta entre 0 et 2pi : ", len(theta))
    ordre_fourier = 5
    print("Nombre de mode de fourier {} \n".format(ordre_fourier))

    #GENERATION
    print("PARAMETRES GENERATION : ")
    nb_points_generation = 150
    t_anevrisme = np.linspace(0, 1, nb_points_generation)
    print("Nombre de points reconstruction centerline générée : ", nb_points_generation)
    print("Nombre d'anévrismes générés : ", generation)
    print("-------------------------------------------------------------------")
    ############################################################################
    ############################################################################

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ LECTURE DES DOSSIERS -----------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    start_lecture = time.time()
    liste_centerline = sorted(os.listdir(doss_centerline))
    liste_stl = sorted(os.listdir(doss_mesh))

    liste_control = []
    liste_mesh = []
    for i in range(len(liste_centerline)):
        CONTROL = gf.Lecture_csv_pcl(doss_centerline + liste_centerline[i])
        liste_control.append(CONTROL)
        mesh = trimesh.load_mesh(doss_mesh + liste_stl[i])
        liste_mesh.append(mesh)
    lecture_time = time.time()
    print("Temps d'importation des données : ", round(lecture_time - start_lecture, 3))

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ CREATION BASE CENTERLINE -------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    BASE_CENTERLINE = espace.Creation_Base_Main.Main_base_centerline(liste_control, nb_points_reconstruction, degree_reconstruction)
    coeff_dilatation = espace.Base_Centerline_Utilities.Dilatation(liste_control)
    gf.Write_csv("resultats/base/BASE_CENTERLINE.csv", BASE_CENTERLINE, "")

    center_time = time.time()
    print("Temps création base centerline : ", round(center_time - lecture_time, 3))

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ CREATION BASE CONTOUR ET RAYON -------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    BASE_CONTOUR, BASE_RAYON = espace.Creation_Base_Main.Main_base_contour(liste_control, liste_mesh, liste_t, degree_reconstruction, ordre_fourier)
    gf.Write_csv("resultats/base/BASE_CONTOUR.csv", BASE_CONTOUR, "")
    gf.Write_csv("resultats/base/BASE_RAYON.csv", BASE_RAYON, "")

    contour_time = time.time()
    print("Temps création base contour et rayon : ", round(contour_time - center_time, 3))

    generation_time = time.time()
    if generation != 0 :

        print('---------------------------------------------------------------------------------------------------------------------------')
        print('------------------------------------------- GENERATION DES ANEVRISMES ALEATOIRES ------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------------------------')

        for nb in range(generation):

            ANEVRISME = espace.Creation_Base_Main.Main_Generation(BASE_CENTERLINE, BASE_CONTOUR, BASE_RAYON, coeff_dilatation, t_anevrisme)
            gf.Write_csv("resultats/classe_anevrismes/ANEVRISME_{}.csv".format(nb), ANEVRISME, "x, y, z, label")

    end_time = time.time()
    print("Temps pour la génération de {} anévrismes : {} ". format(generation, round(end_time - generation_time, 3)))
    print("Temps total de la méthode : ", round(end_time - start_time, 3))
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ FIN ESPACE ANEVRISMES ----------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    return 0

def Parametrisation(file_control, file_stl):
    """ Fonction permettand d'effectuer la reconstruction d'une géométrie grâce à la centerline et la géométrie initiale."""

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ DEBUT PARAMETRISATION ----------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    start_time = time.time()

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ LECTURE DES DONNEES ------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    CONTROL = gf.Lecture_csv_pcl(file_control)
    aorte = trimesh.load_mesh(file_stl)

    lecture_time = time.time()
    print("Temps pour lecture des données : ", round(lecture_time - start_time, 3))

    ############################################################################
    ######################## PARAMETRES UTILISES ###############################
    print("-------------------------- PARAMETRES -----------------------------")
    liste_t = np.linspace(0, 1, 200)
    print("Nombre de coupures : ", len(liste_t))
    liste_theta = np.linspace(0, 2*np.pi, 150)
    print("Nombre de theta entre 0 et 2pi : ", len(liste_theta))
    degree = 3
    print("Le degré des BSplines est de : ", degree)
    coeff_fourier = 5
    print("Modes de la série de fourier : ", coeff_fourier)
    print("-------------------------------------------------------------------")
    ############################################################################
    ############################################################################

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ PARAMETRISATION ----------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')
    param_time = time.time()

    COORDONNEES, PAROI = espace.Parametrisation_Main.Parametrisation(CONTROL, aorte, liste_t, liste_theta, degree, coeff_fourier)

    gf.Write_csv("resultats/parametrisation/CENTERLINE.csv", COORDONNEES, "x, y, z")
    gf.Write_csv("resultats/parametrisation/PAROI.csv", PAROI, "x, y, z")

    end_time = time.time()
    print("Temps paramétrisation : ", round(end_time - param_time, 3))
    print("Temps total  méthode : ", round(end_time - start_time, 3))

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ FIN PARAMETRISATION ------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    return 0
