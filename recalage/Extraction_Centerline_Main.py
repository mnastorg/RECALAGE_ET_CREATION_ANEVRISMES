###################################################################################################################################
###################################### MODULE POUR EXTRACTION DU MORCEAU DE LA CENTERLINE QUI NOUS INTERESSE ######################
###################################################################################################################################

import numpy as np
import time

from outils import Gestion_Fichiers as gf
from outils import BSplines_Utilities as bs

gf.Reload(bs)

def Main_Extraction_Centerline_Labels(new_bsplines, liste_label, nb_control = 5, degree = 3):
    """ Fonction permettant d'extraire un bout de la centerline correspondant aux labels donnés
    dans liste_label."""

    start_time = time.time()

    centerline = []
    BSPLINES = np.vstack(new_bsplines)

    for i in liste_label :
        LAB = BSPLINES[BSPLINES[:,3] == i]
        centerline.append(LAB)

    TOTAL = np.vstack(centerline)
    nb_points = np.shape(TOTAL)[0]

    T = np.linspace(0, nb_points-1, nb_control, dtype = 'int')
    t = np.linspace(0, 1, nb_points)

    CONTROL = TOTAL[T,0:3]

    KNOT = bs.Knotvector(CONTROL, degree)

    BASIS = bs.Matrix_Basis_Function(degree, KNOT, np.shape(CONTROL)[0], t, 0)
    BASIS_DER1 = bs.Matrix_Basis_Function(degree, KNOT, np.shape(CONTROL)[0], t, 1)
    BASIS_DER2 = bs.Matrix_Basis_Function(degree, KNOT, np.shape(CONTROL)[0], t, 2)

    #CALCUL DES BSPLINES ET AJOUT DES LABELS
    BSPLINES = np.dot(BASIS, CONTROL)

    centerline_parametres = [CONTROL, BSPLINES, KNOT, BASIS, BASIS_DER1, BASIS_DER2, t]

    end_time = time.time()
    print("Temps pour étape 3 : Création de la courbe centrale", round(end_time - start_time, 2))

    return TOTAL[:,0:3], centerline_parametres
