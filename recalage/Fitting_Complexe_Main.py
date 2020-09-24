###################################################################################################################################
###################################### MODULE POUR LE RECALAGE D'UNE GEOMETRIE COMPLEXE ###########################################
###################################################################################################################################

import numpy as np
import time
from math import *
import matplotlib.pyplot as plt

from outils import Gestion_Fichiers as gf
from recalage import Initialisation_Utilities as init
from recalage import Fitting_Complexe_Utilities as ftcu
from recalage import Connectivite_Utilities as connect

gf.Reload(ftcu)
gf.Reload(connect)
gf.Reload(init)

def Main_fitting_complexe(PCL, liste_init, degree = 2, max_iter = 5, tol = 1.e-4, rig = 1.e-6):
    """Procède au fitting d'une géométrie complexe i.e enchainement de BSplines à reconnecter."""

    debut = time.time()

    #EXTRACTIONS DES ELEMENTS INITIAUX
    liste_controle = liste_init[0]
    liste_bsplines = liste_init[1]
    liste_knot = liste_init[2]
    liste_basis_der2 = liste_init[3]
    liste_t = liste_init[4]
    merge = liste_init[5]

    liste_nb_controle, nb_controle = init.Nb_points_liste(liste_controle)

    print("Il y a {} points de controle, donc un vecteur inconnu de taille {}".format(nb_controle, 3*nb_controle))

    tab_it = []
    tab_err = [0,1]
    e = []
    it = 0

    while it < max_iter and np.abs(tab_err[-1] - tab_err[-2]) > tol :

        print("Itération numéro : ", it )
        start_time = time.time()

        #FITTING SIMPLE
        new_control, new_bsplines, erreur = ftcu.Fitting(PCL, liste_controle, liste_bsplines, liste_knot, liste_basis_der2, degree, liste_t, rig)
        print("Erreur est de : ", round(erreur, 3))
        tab_err.append(erreur)
        e.append(erreur)

        #MERGE DES POINTS DE CONNECTION
        merge_control, merge_bsplines = connect.Process_connection(new_control, liste_knot, degree, liste_t, merge)

        #PROJECTION DES POINTS SAILLANTS
        proj_control, proj_bsplines, proj_der1, proj_der2 = connect.Projection(merge_bsplines, merge_control, liste_knot, degree, liste_t)

        #MISE A JOUR
        liste_controle = proj_control
        liste_bsplines = proj_bsplines

        #INCREMENTATION
        it += 1

        end_time = time.time()
        print("Temps de calcul : ", round(end_time - start_time, 2))

    plt.plot(np.arange(it), e, '-o')
    plt.title("Erreur en fonction du nombre d'itérations")
    plt.show()

    return liste_controle, liste_bsplines, proj_der1, proj_der2
