import numpy as np
from math import *

from scipy.spatial import distance
from scipy import optimize
from scipy import linalg

import Gestion_Fichiers as gf
import BSplines_Utilities as bs
import Initialisation_Utilities as init

gf.Reload(bs)
gf.Reload(init)

###########################################################################################################################################
#################################################### FONCTIONS ASSOCIEES AUX FOOT POINTS ##################################################
###########################################################################################################################################

def Fitting(PCL, liste_controle, liste_bsplines, liste_knot, liste_basis_der2, degree, liste_t, rig):
    """Fonction effectuant le fitting entre le PCL et la liste_bsplines. En particulier, on déplace
    les points de controles de manière optimale afin que l'intégralité des bsplines fit au mieux
    le nuage de point PCL. L'algorithme minimise un 'double' probleme des moindres carres de la forme
    ||Ax - b||**2 + rig*||Bx - c||."""

    ########################################################################################################################################
    ############ FITTING EN 4 PARTIES ######################################################################################################
    ########################################################################################################################################

    ########################################################################################################################################
    ############ VARIABES PRELIMINAIRES ####################################################################################################
    ########################################################################################################################################

    #CALCUL DU NOMBRE TOTAL DE POINTS DE CONTROL ET LE
    #NOMBRE DE POINTS DE CONTROL PAR LABEL

    nb_control, liste_nb_control = Nb_element_liste_array(liste_controle)
    CONTROL = np.vstack(liste_controle)
    BSPLINES = np.vstack(liste_bsplines)
    controle = Matrix_to_vector(CONTROL[:,0:3], dimension = 3)

    ########################################################################################################################################
    ########### ETAPE 1 - ON CALCULE LA MATRICE DE FOOT POINT ET LA MATRICE BASE ASSOCIEE ##################################################
    ########################################################################################################################################

    #CALCUL VECTEUR INDICE DES FOOTPOINTS
    index_foot_point = Foot_Point(PCL, BSPLINES[:,0:3])
    #ON CREE LA MATRICE DES POINTS DE BSPLINES CORRESPONDANT A CELUI DU NUAGE
    BSPLINES_FOOT_POINT = BSPLINES[index_foot_point]
    #CALCUL DE LA MATRICE DE BASE DES FOOTPOINTS
    BASIS, BASIS_DER1, BASIS_DER2 = Matrix_Basis_Foot_Point(BSPLINES_FOOT_POINT, liste_controle, liste_bsplines, liste_knot, degree, liste_t)

    ########################################################################################################################################
    ########### ETAPE 2 - CALCUL DU 2ND MEMBRE ET DE LA MATRICE BLOCK ######################################################################
    ########################################################################################################################################

    #CALCUL DE LA MATRICE BLOCK A
    BIG_BASIS = linalg.block_diag(BASIS, BASIS, BASIS)
    #CALCUL DU SECOND MEMBRE b
    SND_OBJ = Matrix_to_vector(PCL - BSPLINES_FOOT_POINT[:,0:3], dimension = 3)

    #CALCUL DE LA MATRICE BLOCK B
    BIG_BASIS_DER2 = linalg.block_diag(linalg.block_diag(*liste_basis_der2), linalg.block_diag(*liste_basis_der2), linalg.block_diag(*liste_basis_der2))
    #CALCUL DU SECOND MEMBRE c
    SND_REG = np.dot(BIG_BASIS_DER2, controle)

    ########################################################################################################################################
    ########### ETAPE 3 - ALGORITHME DE MINIMISATION #######################################################################################
    ########################################################################################################################################

    D0 = np.zeros(3*nb_control)
    #D = optimize.minimize(ToMinimize_3D, D0, args = (BIG_BASIS, BIG_BASIS_DER2, SND_OBJ, SND_REG, rig), method = 'SLSQP', jac = Gradient)
    D = optimize.minimize(ToMinimize_3D, D0, args = (BIG_BASIS, BIG_BASIS_DER2, SND_OBJ, SND_REG, rig), method = 'Newton-CG', jac = Gradient, hess = Hessienne)

    ########################################################################################################################################
    ########### ETAPE 4 - MISE A JOUR DES BSPLINES PAR VECTEUR D ###########################################################################
    ########################################################################################################################################

    new_control, new_bsplines = BSplines_Update_3D(D.x, liste_controle, liste_knot, degree, liste_t)
    nb_points, liste_points = Nb_element_liste_array(new_bsplines)
    erreur  = D.fun/(3*nb_points)

    return new_control, new_bsplines, erreur

############################################################################################################################################
#################################################### FONCTIONS UTILES ######################################################################
############################################################################################################################################

def Nb_element_liste_array(liste_array):
    """Fonction permettant de donner le nombre de points total
    ainsi qu'une liste de nombre de points par indice"""
    nb = []
    for i in range(len(liste_array)):
        nb.append(np.shape(liste_array[i])[0])
    somme = np.sum(nb)

    return somme, nb

def Matrix_to_vector(MATRIX, dimension = 3):
    """Transforme un tableau [x,y,z] en un vecteur de taille
    3x nblignes du tableau et prenant d'abord x puis y puis z"""
    return np.reshape(MATRIX, dimension*np.shape(MATRIX)[0], order = 'F')[np.newaxis].T

############################################################################################################################################
############################################## FONCTIONS ASSOCIEES AUX FOOT POINTS #########################################################
############################################################################################################################################

def Foot_Point(PCL, BSPLINES):
    """ Retourne un vecteur d'indices correspondants aux foot points.
    Plus précisément, on associe à tout point du nuage son point d'une
    BSplines équivalents"""

    DIST = distance.cdist(PCL, BSPLINES)
    minimum = np.argmin(DIST, axis = 1)

    return minimum

def Matrix_Basis_Foot_Point(BSPLINES_FOOT_POINT, liste_controle, liste_bsplines, liste_knot, degree, liste_t):
    """ Fonction retournant la matrice de base associé au Foot_Point. Pour un élément dans Foot Point, on extrait son
    label. Puis on cherche dans le label correspondant de liste_bsplines le point équivalent. On recupere ainsi
    toutes les informations dont on a besoin pour créer la fonction de base, pour chaque élément du foot point."""

    nb_control, liste_shape = Nb_element_liste_array(liste_controle)
    cumul_sum = np.cumsum(liste_shape)

    nb_foot_point = np.shape(BSPLINES_FOOT_POINT)[0]

    BASIS = np.zeros((nb_foot_point,nb_control))
    BASIS_DER1 = 0*BASIS
    BASIS_DER2 = 0*BASIS

    #ON PARCOURT LES FOOTPOINTS
    for i in range(nb_foot_point):
        #ON REGARDE LE LABEL DE LA LIGNE i
        lab = int(BSPLINES_FOOT_POINT[i,3])
        #ON SELECTIONNE LA BSPLINES CORRESPONDANT A LA BRANCHE i
        TAB_TO_SEARCH = (liste_bsplines[lab])[:,0:3]
        #ON SELECTIONNE LA LIGNE DONT ON VEUT TROUVER l'INDICE
        search = BSPLINES_FOOT_POINT[i,0:3]
        #ON CHERCHE L'INDICE DANS TAB_TO_SEARCH DE search. CELA NOUS DONNERA UNE
        #INDICATION SUR LE 't' CORRESPONDANT DANS NEW_BSPLINES
        index = int( (np.where(np.all(TAB_TO_SEARCH == search, axis = 1)))[0] )

        if (lab == 0):
            for j in range(liste_shape[lab]):
                BASIS[i,j] = bs.Basis_Function_Der(degree, liste_knot[lab], j, (liste_t[lab])[index], 0)
                BASIS_DER1[i,j] = bs.Basis_Function_Der(degree, liste_knot[lab], j, (liste_t[lab])[index], 1)
                BASIS_DER2[i,j] = bs.Basis_Function_Der(degree, liste_knot[lab], j, (liste_t[lab])[index], 2)

        else :
            #ICI CUMUL SUM EST POUR PLACER AU BON ENDROIT LES POINTS DE CONTROLE PAR RAPPORT AUX LABELS
            debut = cumul_sum[lab-1]
            for j in range(liste_shape[lab]):
                BASIS[i,debut+j] = bs.Basis_Function_Der(degree, liste_knot[lab], j, (liste_t[lab])[index],0)
                BASIS_DER1[i,debut+j] = bs.Basis_Function_Der(degree, liste_knot[lab], j, (liste_t[lab])[index], 1)
                BASIS_DER2[i,debut+j] = bs.Basis_Function_Der(degree, liste_knot[lab], j, (liste_t[lab])[index], 2)

    return BASIS, BASIS_DER1, BASIS_DER2

###########################################################################################################################################
############################################### FONCTIONS POUR LA MINIMISATION ############################################################
###########################################################################################################################################

def ToMinimize_3D(D, BIG_BASIS, BIG_BASIS_DER2, SND_OBJ, SND_REG, rig):

    f_obj = 0.5*np.linalg.norm( np.dot(BIG_BASIS , D[np.newaxis].T) - SND_OBJ )**2

    NOR_DER2 = 0.5*np.linalg.norm( np.dot(BIG_BASIS_DER2, D[np.newaxis].T) + SND_REG )**2

    return f_obj + rig*NOR_DER2

def Gradient(D, BIG_BASIS, BIG_BASIS_DER2, SND_OBJ, SND_REG, rig):

    PART1 = np.dot(BIG_BASIS.T, np.dot(BIG_BASIS , D[np.newaxis].T) - SND_OBJ)

    PART2 = rig*( np.dot(BIG_BASIS_DER2.T , np.dot(BIG_BASIS_DER2, D[np.newaxis].T) + SND_REG) )

    return np.squeeze(PART1 + PART2)

def Hessienne(D, BIG_BASIS, BIG_BASIS_DER2, SND_OBJ, SND_REG, rig):

    return np.dot(BIG_BASIS.T, BIG_BASIS) + rig*np.dot(BIG_BASIS_DER2.T, BIG_BASIS_DER2)

############################################################################################################################################
############################################## FONCTIONS MISE A JOUR #######################################################################
############################################################################################################################################

def BSplines_Update_3D(D, liste_controle, knot, degree, liste_t):
    """ Ressort les nouveaux points de controles ainsi que les nouvelles bsplines apres
    Update par le vecteur D"""

    nb_control, liste_shape = Nb_element_liste_array(liste_controle)

    new_bsplines = []
    new_control = []

    n_precedent = 0

    #ON PARCOURT LES BRANCHES
    for i in range(len(liste_controle)):

        #SI ON EST SUR LA PREMIERE BRANCHE
        if i == 0 :

            #CONTROL POINTS ET KNOT
            CONTROL = liste_controle[i]
            KNOT = knot[i]

            #ON REMET LES ANCIENS POINTS DE CONTROLES ET BSPLINES SI PETITES BRANCHES
            if len(liste_t[i]) <= 30 :
                NEW_BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(CONTROL[:,0:3], KNOT, degree, liste_t[i], dimension = 3)
                NEW_BSPLINES = np.insert(NEW_BSPLINES, 3, CONTROL[0,3], axis = 1)
                NEW_BSPLINES = init.Ajout_saillants_leafs_bsplines(NEW_BSPLINES, CONTROL)
                new_bsplines.append(NEW_BSPLINES)
                new_control.append(CONTROL)

            else :
                #ON ORGANISE LE VECTEUR DE MAJ
                n_actuel = liste_shape[i]
                DD = np.zeros((n_actuel,3))
                DD[:,0] = D[0:n_actuel]
                DD[:,1] = D[nb_control : nb_control+n_actuel]
                DD[:,2] = D[2*nb_control : 2*nb_control+n_actuel]

                #ON CREE LA NOUVELLE BSPLINES
                PPLUS = CONTROL[:,0:3] + DD
                PPLUS = np.insert(PPLUS, 3, CONTROL[:,3], axis = 1)
                PPLUS = np.insert(PPLUS, 4, CONTROL[:,4], axis = 1)
                PPLUS = np.insert(PPLUS, 5, CONTROL[:,5], axis = 1)
                NEW_BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(PPLUS[:,0:3], KNOT, degree, liste_t[i], dimension = 3)
                NEW_BSPLINES = np.insert(NEW_BSPLINES, 3, PPLUS[0,3], axis = 1)
                NEW_BSPLINES = init.Ajout_saillants_leafs_bsplines(NEW_BSPLINES, PPLUS)

                new_bsplines.append(NEW_BSPLINES)
                new_control.append(PPLUS)

        #SI ON EST SUR LES AUTRES BRANCHES
        else:

            #CONTROL POINTS ET KNOT
            CONTROL = liste_controle[i]
            KNOT = knot[i]

            #ON REMET LES ANCIENS POINTS DE CONTROLES ET BSPLINES SI PETITES BRANCHES
            if len(liste_t[i]) <= 30 :
                n_precedent += liste_shape[i-1]
                NEW_BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(CONTROL[:,0:3], KNOT, degree, liste_t[i], dimension = 3)
                NEW_BSPLINES = np.insert(NEW_BSPLINES, 3, CONTROL[0,3], axis = 1)
                NEW_BSPLINES = init.Ajout_saillants_leafs_bsplines(NEW_BSPLINES, CONTROL)

                new_bsplines.append(NEW_BSPLINES)
                new_control.append(CONTROL)

            #SI C'EST UNE GRANDE BRANCHE
            else :
                #ON ORGANISE LE VECTEUR DE MISE A JOUR
                n_precedent += liste_shape[i-1]
                n_actuel = liste_shape[i]
                DD = np.zeros((n_actuel,3))
                DD[:,0] = D[n_precedent: n_precedent + n_actuel]
                DD[:,1] = D[nb_control + n_precedent : nb_control + n_precedent + n_actuel]
                DD[:,2] = D[2*nb_control + n_precedent : 2*nb_control + n_precedent + n_actuel]

                #ON CREE LA NOUVELLE BSPLINES
                PPLUS = CONTROL[:,0:3] + DD
                PPLUS = np.insert(PPLUS, 3, CONTROL[:,3], axis = 1)
                PPLUS = np.insert(PPLUS, 4, CONTROL[:,4], axis = 1)
                PPLUS = np.insert(PPLUS, 5, CONTROL[:,5], axis = 1)
                NEW_BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(PPLUS[:,0:3], KNOT, degree, liste_t[i], dimension = 3)
                NEW_BSPLINES = np.insert(NEW_BSPLINES, 3, PPLUS[0,3], axis = 1)
                NEW_BSPLINES = init.Ajout_saillants_leafs_bsplines(NEW_BSPLINES, PPLUS)

                new_bsplines.append(NEW_BSPLINES)
                new_control.append(PPLUS)

    return new_control, new_bsplines
