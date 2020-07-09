import numpy as np
from math import *

from scipy.spatial import distance
from scipy import optimize
from scipy import linalg

import Gestion_Fichiers as gf
import BSplines_Utilities as bs
import Initialisation_Utilities as init
import Statistiques_Utilities as stats

gf.Reload(bs)
gf.Reload(init)

###########################################################################################################################################
#################################################### FONCTIONS ASSOCIEES AUX FOOT POINTS ##################################################
###########################################################################################################################################

def Fitting(PCL, CONTROL, BSPLINES, KNOT, BASIS_DER2, degree, t, rig):
    """Fonction effectuant le fitting entre le PCL et une courbe BSPLINES initiale. En particulier, on déplace
    les points de controles de manière optimale afin la bsplines 'fit' au mieux le nuage de point PCL.
    L'algorithme minimise un 'double' probleme des moindres carres de la forme ||Ax - b||**2 + rig*||Bx - c||."""

    ########################################################################################################################################
    ############ FITTING EN 4 PARTIES ######################################################################################################
    ########################################################################################################################################

    ########################################################################################################################################
    ############ VARIABES PRELIMINAIRES ####################################################################################################
    ########################################################################################################################################

    #CALCUL DU NOMBRE TOTAL DE POINTS DE CONTROL
    nb_control  = np.shape(CONTROL)[0]
    controle = Matrix_to_vector(CONTROL, dimension = 3)

    ########################################################################################################################################
    ########### ETAPE 1 - ON CALCULE LA MATRICE DE FOOT POINT ET LA MATRICE BASE ASSOCIEE ##################################################
    ########################################################################################################################################

    #CALCUL VECTEUR INDICE DES FOOTPOINTS
    index_foot_point = Foot_Point(PCL, BSPLINES)
    #ON CREE LA MATRICE DES POINTS DE BSPLINES CORRESPONDANT A CELUI DU NUAGE
    BSPLINES_FOOT_POINT = BSPLINES[index_foot_point]
    #CALCUL DE LA MATRICE DE BASE DES FOOTPOINTS
    FP_BASIS, FP_BASIS_DER1, FP_BASIS_DER2 = Matrix_Basis_Foot_Point(BSPLINES_FOOT_POINT, CONTROL, BSPLINES, KNOT, degree, t)

    ########################################################################################################################################
    ########### ETAPE 2 - CALCUL DU 2ND MEMBRE ET DE LA MATRICE BLOCK ######################################################################
    ########################################################################################################################################

    #CALCUL DE LA MATRICE BLOCK A
    BIG_BASIS = linalg.block_diag(FP_BASIS, FP_BASIS, FP_BASIS)
    #CALCUL DU SECOND MEMBRE b
    SND_OBJ = Matrix_to_vector(PCL - BSPLINES_FOOT_POINT, dimension = 3)

    #CALCUL DE LA MATRICE BLOCK B
    BIG_BASIS_DER2 = linalg.block_diag(linalg.block_diag(FP_BASIS_DER2), linalg.block_diag(FP_BASIS_DER2), linalg.block_diag(FP_BASIS_DER2))
    #CALCUL DU SECOND MEMBRE c
    SND_REG = np.dot(BIG_BASIS_DER2, controle)

    ########################################################################################################################################
    ########### ETAPE 3 - ALGORITHME DE MINIMISATION #######################################################################################
    ########################################################################################################################################

    D0 = np.zeros(3*nb_control)

    cons = ({'type' : 'eq', 'fun' : lambda D : (np.reshape(D, (int(len(D)/3) , 3), order = 'F'))[0,:]},
            {'type' : 'eq', 'fun' : lambda D : (np.reshape(D, (int(len(D)/3) , 3), order = 'F'))[-1,:]})
    #D = optimize.minimize(ToMinimize_3D, D0, args = (BIG_BASIS, BIG_BASIS_DER2, SND_OBJ, SND_REG, rig), method = 'SLSQP', jac = Gradient)
    D = optimize.minimize(ToMinimize_3D, D0, args = (BIG_BASIS, BIG_BASIS_DER2, SND_OBJ, SND_REG, rig), method = 'SLSQP', jac = Gradient, constraints = cons)

    ########################################################################################################################################
    ########### ETAPE 4 - MISE A JOUR DES BSPLINES PAR VECTEUR D ###########################################################################
    ########################################################################################################################################

    NEW_CONTROL, NEW_BSPLINES = BSplines_Update_3D(D.x, CONTROL, KNOT, degree, t)
    nb_points = np.shape(NEW_BSPLINES)[0]
    erreur  = D.fun/(3*nb_points)

    return NEW_CONTROL, NEW_BSPLINES, erreur

############################################################################################################################################
#################################################### FONCTIONS UTILES ######################################################################
############################################################################################################################################

def Matrix_to_vector(MATRIX, dimension = 3):
    """Transforme un tableau [x,y,z] en un vecteur de taille
    3x nblignes du tableau et prenant d'abord x puis y puis z"""
    return np.reshape(MATRIX, dimension*np.shape(MATRIX)[0], order = 'F')[np.newaxis].T

def Vector_to_matrix(vector, dimension = 3):
    return np.reshape(vector, (int(len(vector)/dimension) , dimension), order = 'F')

############################################################################################################################################
############################################## FONCTIONS ASSOCIEES AUX FOOT POINTS #########################################################
############################################################################################################################################

def Foot_Point(PCL, BSPLINES):
    """ Retourne un vecteur d'indices correspondants aux foot points.
    Plus précisément, on associe à tout point du nuage son point son
    équivalent dans la Bsplines."""

    DIST = distance.cdist(PCL, BSPLINES)
    minimum = np.argmin(DIST, axis = 1)

    return minimum

def Matrix_Basis_Foot_Point(BSPLINES_FOOT_POINT, CONTROL, BSPLINES, KNOT, degree, t):
    """ Fonction retournant la matrice de base associé au Foot_Point."""

    nb_control = np.shape(CONTROL)[0]

    nb_foot_point = np.shape(BSPLINES_FOOT_POINT)[0]

    BASIS = np.zeros((nb_foot_point, nb_control))
    BASIS_DER1 = 0*BASIS
    BASIS_DER2 = 0*BASIS

    #ON PARCOURT LES FOOTPOINTS
    for i in range(nb_foot_point):
        #ELEMENT A CHERCHER DANS LA MATRICE DES BSPLINES
        search = BSPLINES_FOOT_POINT[i,:]
        #ON CHERCHE L'INDICE DANS BSPLINES DE search. CELA NOUS DONNERA UNE
        #INDICATION SUR LE 't' CORRESPONDANT DANS NEW_BSPLINES
        index = int( (np.where(np.all(BSPLINES == search, axis = 1)))[0] )

        for j in range(nb_control):
            BASIS[i,j] = bs.Basis_Function_Der(degree, KNOT, j, t[index], 0)
            BASIS_DER1[i,j] = bs.Basis_Function_Der(degree, KNOT, j, t[index], 1)
            BASIS_DER2[i,j] = bs.Basis_Function_Der(degree, KNOT, j, t[index], 2)

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

def BSplines_Update_3D(D, CONTROL, KNOT, degree, t):
    """ Ressort les nouveaux points de controles ainsi que les nouvelles bsplines apres
    Update par le vecteur D"""

    nb_control = np.shape(CONTROL)[0]

    DD = Vector_to_matrix(D, dimension = 3)

    #ON CREE LA NOUVELLE BSPLINES
    NEW_CONTROL = CONTROL + DD
    NEW_CONTROL[0,:] = CONTROL[0,:]
    NEW_CONTROL[-1,:] = CONTROL[-1,:]

    NEW_BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(NEW_CONTROL, KNOT, degree, t, dimension = 3)

    return NEW_CONTROL, NEW_BSPLINES

def BSplines_Finale(CONTROL, KNOT, degree, t):

    BSPLINES, DER1, DER2, DER3 = bs.BSplines_RoutinePython(CONTROL, KNOT, degree, t, dimension = 3)

    TAN = stats.tangente(DER1)
    BI = stats.binormale(DER1, DER2)
    NOR = stats.normale(BI, TAN)

    courbure = stats.courbure(DER1, DER2)

    TAB = np.hstack((BSPLINES, TAN, NOR, BI, courbure))

    return TAB
