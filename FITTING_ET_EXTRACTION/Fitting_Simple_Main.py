import numpy as np
import time

import Gestion_Fichiers as gf
import Fitting_Simple_Utilities as ftsu

gf.Reload(gf)
gf.Reload(ftsu)

def Main_fitting_simple(PCL, liste_init, degree = 5, max_iter = 100, tol = 1.e-8, rig = 0):
    """Procède au fitting d'une géométrie complexe i.e enchainement de BSplines à reconnecter."""

    #EXTRACTIONS DES ELEMENTS INITIAUX
    CONTROL = liste_init[0]
    BSPLINES = liste_init[1]
    KNOT = liste_init[2]
    BASIS_DER2 = liste_init[3]
    t = liste_init[4]

    tab_it = []
    tab_err = [0,1]
    it = 0

    start_time = time.time()
    print("Max_iterations = ", max_iter)
    print("Tolérance erreur = ", tol)
    print("Rigidité = ", rig)

    while it < max_iter and np.abs(tab_err[-1] - tab_err[-2]) > tol :

        #FITTING SIMPLE
        NEW_CONTROL, NEW_BSPLINES, erreur = ftsu.Fitting(PCL, CONTROL, BSPLINES, KNOT, BASIS_DER2, degree, t, rig)
        tab_err.append(erreur)

        #MISE A JOUR
        CONTROL = NEW_CONTROL
        BSPLINES = NEW_BSPLINES

        #INCREMENTATION
        it += 1

    print("Nombre d'itérations effectués = ", it)
    print("Erreur finale = ", erreur)

    BSPLINES = ftsu.BSplines_Finale(CONTROL, KNOT, degree, t)
    
    end_time = time.time()
    print("Temps de calcul : ", round(end_time - start_time, 2))

    return CONTROL, BSPLINES
