import numpy as np
import time

import Gestion_Fichiers as gf
import Initialisation_Utilities as init

gf.Reload(init)

def Main_Initialisation(NUREA, degree):
    """ Fonction permettant d'initialiser le squelette à fitting. Il agit en 3 étapes.
    La premiereest une étape de Reconstruction brute servant uniquement à calculer le
    repère TNB. La seconde transforme une bifurcation d'1 point en autant de points qu'il
    y a de branches qui arrivent sur la bifurcation. Puis elle sort un nombre de points de
    controle par branche déterminé par rapport à la courbure. Finalement la dernière étape
    détermine les connectivités au niveau des bifurcation. On exprime les points de contrôle
    qui doivent se connecter et ceux qui doivent se projeter (les saillants)."""


    start_time = time.time()

    #### ETAPE 1 - TABLEAU DE RECONSTRUCTION
    RECONSTRUCTION = init.Reconstruction(NUREA)
    reconstruction_time = time.time()
    print("Temps pour étape 1 : Reconstruction = ", round(reconstruction_time - start_time, 2))

    #LISTE DE LINSPACE POUR AVOIR LE MEME NOMBRE DE POINTS QUE LE SQUELETTE DE NUREA
    liste_t = init.Liste_pour_linspace(RECONSTRUCTION)

    #### ETAPE 2 - ON CREE LES CONTROLES AVEC SAILLANTS MULTIPLES AUX BIFURCATIONS ET BSPLINES INITIALES AVEC TNB
    controle_init = init.Creation_controles(RECONSTRUCTION)
    knot_init, basis_init, basis_der1_init, basis_der2_init, bsplines_init, der1_init, der2_init = init.Creation_bsplines(controle_init, degree, liste_t)
    init_time = time.time()
    print("Temps pour étape 2 : Initialisation = ", round(init_time - reconstruction_time, 2))

    #### ETAPE 3 - ON RECONSTRUIT LES POINTS DE CONTROLE AVEC LA NOTION DE SAILLANT "EXACT" ET CONNECTIVITE ENTRE LE BRANCHES
    liste_controle, merge = init.Connectivite(controle_init, bsplines_init)
    liste_knot, liste_basis, liste_basis_der1, liste_basis_der2, liste_bsplines, liste_der1, liste_der2 = init.Creation_bsplines(liste_controle, degree, liste_t)
    connectivite_time = time.time()
    print("Temps pour étape 3 : Connectivite = ", round(connectivite_time - init_time, 2))

    end_time = time.time()
    print("Temps total des trois étapes = ", round(end_time - start_time, 2))


    return RECONSTRUCTION, liste_controle, liste_bsplines, liste_knot, liste_basis_der2, liste_t, merge
