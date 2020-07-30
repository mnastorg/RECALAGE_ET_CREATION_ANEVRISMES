###################################################################################################################################
###################################### MODULE POUR GERER LES CONNECTIVITES DANS LES GEOMETRIES COMPLEXES ##########################
###################################################################################################################################

import numpy as np

from scipy.spatial import distance

from outils import BSplines_Utilities as bs
from recalage import Initialisation_Utilities as init

def Process_connection(new_control, liste_knot, degree, liste_t, merge):
    """Fonction qui permet de merger les points selon la liste merge. Elle
    reconstruit les nouveaux points de controles et les nouvelles Bsplines
    associées."""

    NEW_CONTROL = np.vstack(new_control)
    label = list(set(NEW_CONTROL[:,3]))
    label = [int(i) for i in label]

    #BARYCENTRE DES POINTS A MERGER
    for i in merge:
        NEW_CONTROL[i, 0:3] = np.sum(NEW_CONTROL[i,0:3], axis = 0)/len(i)

    merge_controle = []
    for i in label:
        merge_controle.append(NEW_CONTROL[NEW_CONTROL[:,3] == i])

    merge_bsplines = []
    for i in label :
        CONTROL = merge_controle[i]
        MERGE_BSPLINES, MERGE_DER1, MERGE_DER2, MERGE_DER3 = bs.BSplines_RoutinePython(CONTROL[:,0:3], liste_knot[i], degree, liste_t[i], dimension = 3)
        MERGE_BSPLINES = np.insert(MERGE_BSPLINES, 3, CONTROL[0,3], axis = 1)
        MERGE_BSPLINES = init.Ajout_saillants_leafs_bsplines(MERGE_BSPLINES, CONTROL)
        merge_bsplines.append(MERGE_BSPLINES)

    return merge_controle, merge_bsplines

def Projection(new_bsplines, new_control, knot, degree, liste_t):
    """ Fonction permettant de projeter un point saillant sur la BSplines
    la plus proche. """
     
    proj_bsplines = []
    proj_control = []
    proj_der1 = []
    proj_der2 = []
    proj_der3 = []

    BSPLINES = np.vstack(new_bsplines)

    for i in range(len(new_bsplines)):

        CONTROL = new_control[i]
        place = np.where(CONTROL[:,4] == 1)
        SAIL = CONTROL[place]
        NEW = np.delete(BSPLINES, np.where(BSPLINES[:,3] == i), axis = 0)

        DIST = distance.cdist(SAIL[:,0:3], NEW[:,0:3])
        index = np.argmin(DIST, axis = 1)
        CONTROL[place, 0:3] = NEW[index, 0:3]
        proj_control.append(CONTROL)

        PROJ_BSPLINES, PROJ_DER1, PROJ_DER2, PROJ_DER3 = bs.BSplines_RoutinePython(CONTROL[:,0:3], knot[i], degree, liste_t[i], dimension = 3)
        PROJ_BSPLINES = np.insert(PROJ_BSPLINES, 3, (CONTROL)[0,3], axis = 1)
        PROJ_BSPLINES = init.Ajout_saillants_leafs_bsplines(PROJ_BSPLINES, CONTROL)

        proj_bsplines.append(PROJ_BSPLINES)
        proj_der1.append(PROJ_DER1)
        proj_der2.append(PROJ_DER2)
        #proj_der3.append(PROJ_DER3)

        BSPLINES[BSPLINES[:,3]==i , 0:3] = PROJ_BSPLINES[:,0:3]

    return proj_control, proj_bsplines, proj_der1, proj_der2
