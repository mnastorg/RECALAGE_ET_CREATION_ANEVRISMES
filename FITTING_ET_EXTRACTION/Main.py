#ON IMPORTE LES MODULES
import numpy as np
import time
import Gestion_Fichiers as gf
import Initialisation_Main as init
import Nuage_Level_Set_Main as nlsm
import Fitting_Complexe_Main as fitc
import Extraction_Centerline_Main as exmain
import Fitting_Simple_Main as fits

#ON RELOAD LES MODULES SI CHANGEMENTS
gf.Reload(gf)
gf.Reload(init)
gf.Reload(nlsm)
gf.Reload(fitc)
gf.Reload(exmain)
gf.Reload(fits)

#LISTE D'EXTRACTION DES LABELS
p001 = [[19,25], 'p001']
p002 = [[8], 'p002']


def Main(PCL, file_csv, liste_centerline):

    start_time = time.time()

    ####################################################################################################################################
    ################################### CREATION DU NUAGE DE POINT "LEVEL_SET" #########################################################
    ####################################################################################################################################
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('--------------------------------------- CREATION DU NUAGE DE POINTS PAR LEVEL SET -----------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')
    """
    pitch = 0.4
    seuil = 0.7

    PCL, LEVELSET = nlsm.Main_nuage_level_set(file_stl, pitch, seuil)
    """
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('--------------------------------------- FIN CREATION DU NUAGE DE POINTS PAR LEVEL SET -------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')
    ####################################################################################################################################
    ######################################### LECTURE ET INITIALISATION SQUELETTE NUREA ################################################
    ####################################################################################################################################
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------ DEBUT DE L INITIALISATION ------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    degree_fitting = 2

    NUREA = gf.Lecture_csv_nurea(file_csv)

    RECONSTRUCTION, liste_controle, liste_bsplines, liste_knot, liste_basis_der2, liste_t, merge = init.Main_Initialisation(NUREA, degree_fitting)

    gf.Write_csv("RESULTATS/INIT_RECONSTRUCTION.csv", RECONSTRUCTION, "x, y, z, label, is_sail, is_leaf, TX, TY, TZ, NX, NY, NZ, BX, BY, BZ, courbure, longueur")
    CONTROL = np.vstack(liste_controle)
    gf.Write_csv("RESULTATS/INIT_CONTROLES.csv", CONTROL, "x, y, z, label, is_sail, is_leaf")
    BSPLINES = np.vstack(liste_bsplines)
    gf.Write_csv("RESULTATS/INIT_BSPLINES.csv", BSPLINES, "x, y, z, label, is_sail, is_leaf, TX, TY, TZ, NX, NY, NZ, BX, BY, BZ")
    SAIL = BSPLINES[BSPLINES[:,4] == 1]
    gf.Write_csv("RESULTATS/INIT_SAILLANTS.csv", SAIL[:,0:3], "x, y, z")

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('-------------------------------------------- FIN DE L INITIALISATION ------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')
    ####################################################################################################################################
    ######################################################### FITTING ##################################################################
    ####################################################################################################################################
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------- DEBUT DU FITTING ----------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    liste_init1 = [liste_controle, liste_bsplines, liste_knot, liste_basis_der2, liste_t, merge]

    new_control, new_bsplines, new_der1, new_der2 = fitc.Main_fitting_complexe(PCL, liste_init1, degree = degree_fitting, max_iter = 5, tol = 1.e-4, rig = 1.e-6)

    NEW_CONTROL = np.vstack(new_control)
    gf.Write_csv("RESULTATS/FIT_CONTROL.csv", NEW_CONTROL, "x, y, z, label, is_sail, is_leaf")
    NEW_BSPLINES = np.vstack(new_bsplines)
    gf.Write_csv("RESULTATS/FIT_BSPLINES.csv", NEW_BSPLINES, "x, y, z, label, is_sail, is_leaf")
    NEW_SAIL = NEW_BSPLINES[NEW_BSPLINES[:,4] == 1]
    gf.Write_csv("RESULTATS/FIT_SAILL.csv", NEW_SAIL[:,0:3], "x, y, z")

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------ FIN DU FITTING -----------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')
    ####################################################################################################################################
    ################################################# EXTRACTION BRANCHE PRINCIPALE ####################################################
    ####################################################################################################################################
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('-------------------------------------- DEBUT EXTRACTION DE LA COURBE PRINCIPALE -------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    degree_extract = 3
    nb_controle_principale = 5

    PRINCIPALE, centerline_parametres = exmain.Main_Extraction_Centerline_Labels(new_bsplines, liste_centerline[0], nb_control = nb_controle_principale, degree = degree_extract)
    gf.Write_csv("RESULTATS/EXTRACT_PRINCIPALE.csv", PRINCIPALE, "x, y, z")
    gf.Write_csv("RESULTATS/EXTRACT_CONTROL.csv", centerline_parametres[0], "x, y, z")
    gf.Write_csv("RESULTATS/EXTRACT_BSPLINES.csv", centerline_parametres[1], "x, y, z")

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('------------------------------------------- FIN EXTRACTION COURBE PRINCIPALE ----------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')
    ####################################################################################################################################
    ############################################ FITTING/SMOOTHING DE LA COURBE PRINCIPALE #############################################
    ####################################################################################################################################
    print('---------------------------------------------------------------------------------------------------------------------------')
    print('-------------------------------------- DEBUT FITTING / SMOOTHING DE LA SPLINE PRINCIPALE ----------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    #ON EFFECTUE LE FITTING DE LA BSPLINES APPROXIMEE SUR LA BSPLINES PRINCIPALE POUR LA RENDRE LISSE
    liste_init2 = [centerline_parametres[0], centerline_parametres[1], centerline_parametres[2], centerline_parametres[5], centerline_parametres[6]]
    SMOOTH_CONTROL, SMOOTH_BSPLINES = fits.Main_fitting_simple(PRINCIPALE, liste_init2, degree = degree_extract, max_iter = 50, tol = 1.e-5, rig = 0)
    gf.Write_csv("RESULTATS/SMOOTH_CONTROL.csv".format(liste_centerline[1]), SMOOTH_CONTROL, "x, y, z")
    gf.Write_csv("RESULTATS/SMOOTH_BSPLINES.csv", SMOOTH_BSPLINES, "x, y, z, TX, TY, TZ, NX, NY, NZ, BX, BY, BZ, courbure")
    gf.Write_csv("COMPARATIFS/CONTROL_PRINCIPALE_{}.csv".format(liste_centerline[1]), SMOOTH_CONTROL, "x, y, z")

    print('---------------------------------------------------------------------------------------------------------------------------')
    print('---------------------------------------- FIN FITTING / SMOOTHING DE LA SPLINE PRINCIPALE ----------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------')

    end_time = time.time()
    print("Temps total de la méthode : ", round(end_time - start_time, 2))

    return 0



#EXTRACTION DE LA BRANCHE PRINCIPALE
#TAB, branche_principale, centerline_parametres = exmain.Main_Extraction_Centerline_Auto(new_bsplines, new_der1, new_der2, LEVELSET)
#print("Les labels formant la branche principale sont : ", branche_principale[0])
#gf.Write_csv("RESULTATS/EXTRACT_STATS.csv", TAB, "x, y, z, Label, is_sail, is_leaf, TX, TY, TZ, NX, NY, NZ, BX, BY, BZ, courbure, rayon")
#gf.Write_csv("RESULTATS/EXTRACT_PRINCIPALE.csv", branche_principale[1], "x, y, z")
#gf.Write_csv("RESULTATS/EXTRACT_CONTROL.csv", centerline_parametres[0], "x, y, z")
#gf.Write_csv("RESULTATS/EXTRACT_BSPLINES.csv", centerline_parametres[1], "x, y, z")
