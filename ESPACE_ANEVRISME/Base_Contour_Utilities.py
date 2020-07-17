import numpy as np
import time
import time
from math import *
from symfit import parameters, variables, sin, cos, Fit
from sklearn.decomposition import PCA
from scipy.spatial import distance
import trimesh

import Gestion_Fichiers as gf
import BSplines_Utilities as bs
import Statistiques_Utilities as stats

gf.Reload(gf)
gf.Reload(bs)
gf.Reload(stats)

def Matrice_coefficients(liste_control, liste_mesh, liste_t, degree, n_ordre):

    coeffs_fourier = []
    determination = []
    rayon = []

    for control, mesh in zip(liste_control, liste_mesh):
        a0 = []

        for t in liste_t :

            COEFFS, erreur =  Extraction_coeffs_fourier(control, mesh, t, degree, n_ordre)
            coeffs_fourier.append(COEFFS)
            determination.append(erreur)
            a0.append(COEFFS[0])
        rayon.append(np.vstack(a0))

    MAT_PARAM = np.hstack(coeffs_fourier)
    RAYON = np.hstack(rayon)

    return MAT_PARAM, RAYON, determination

def Extraction_coeffs_fourier(CONTROL, mesh, t, degree, n_ordre):

    coeffs_fourier = []

    knot = bs.Knotvector(CONTROL, degree)

    #EXTRAIT LES COORDONNEES DU POINT ET LE REPERE DE FRENET
    COORD, TAN, NOR, BI = Point_de_reference(t, CONTROL, knot, degree)
    X_t = COORD[:,0]
    Y_t = COORD[:,1]
    Z_t = COORD[:,2]
    print(COORD)
    print(TAN)
    print(NOR)
    print(BI)
    #CALCUL DE LA MATRICE DE PASSAGE DE LA BASE CANONIQUE A LA BASE DEFINIE PAR
    #LE REPERE DE FRENET
    PASSAGE = Matrice_de_passage(TAN, NOR, BI)

    #COUPURE DE L'AORTE PAR LE PLAN (N/B). POUR TOUT POINT DE LA COUPURE, RESSORT
    #SON ANGLE / NORMALE (COLONNE 0) ET SA DISTANCE / COORD (COLONNE 2).
    #PERMET D'EXPRIMER UN MODELE VIA UNE SERIE DE FOURIER.
    THETA_R_EXP = Modelisation_contour(PASSAGE, COORD, TAN, mesh)

    #EFFECTUE LE FITTING DU MODELE D'ORDRE n VIA UNE SERIE DE FOURIER
    COEFFS, erreur = Modele_Fourier(THETA_R_EXP, ordre = n_ordre)

    return COEFFS, erreur

##################################################################################################################
############################################## FONCTIONS PRINCIPALES #############################################
##################################################################################################################

def Point_de_reference(t, CONTROL, knot, degree):
    """Extraction des coordonnées ainsi que du repère de Frenet associé au point 't' de la BSpline définie
    par les points CONTROL, le vecteur de noeud knot et son degré."""

    COORD = bs.BSplines_BasisFunction(CONTROL, knot, degree, t, 0)
    DER1 = bs.BSplines_BasisFunction(CONTROL, knot, degree, t, 1)
    DER2 = bs.BSplines_BasisFunction(CONTROL, knot, degree, t, 2)

    TAN = stats.tangente(DER1)
    BI = stats.binormale(DER1, DER2)
    NOR = stats.normale(BI, TAN)

    return COORD, TAN, NOR, BI

def Modelisation_contour(PASSAGE, COORD, TAN, mesh):
    """ Fonction effectuant :
    - La coupure du mesh par le plan orthogonal à la tangente
    - Si plusieurs coupures on isole le cercle aortique
    - Effectue le changement de plan dans la base (N, B, T)
    - Ressort la matrice avec col.0 l'angle d'un point/COORD en col.1
    sa distance/COORD. La matrice est ordonnée suivant l'angle."""

    #ON EFFECTUE LA COUPURE PAR LE PLAN
    COUPURE = Coupure_plan(COORD[0,:], TAN[0,:], mesh)

    #ON ISOLE LE CERCLE CENTRAL
    CERCLE = Isolation_Cercle(COUPURE, COORD)

    #ON FAIT LE CHANGEMENT DE PLAN DANS LA BASE (N, B, T)
    COORD_PLAN, COUPURE_PLAN = Changement_de_plan(PASSAGE, COORD, CERCLE)

    #RESSORT LE TABLEAU DE THETA R
    TAB = Theta_R_Experimental(COORD_PLAN, COUPURE_PLAN)

    return TAB

##################################################################################################################
################################ FONCTIONS COUPURE ET EXTRACTION CERCLE AORTE ####################################
##################################################################################################################

def Coupure_plan(COORD, TAN, mesh):
    """Effectue la coupure dans le plan définit par les 2 vecteurs orthogonaux
    à la tangente (i.e la normale et binormale)."""

    COUPURE = np.vstack(trimesh.intersections.mesh_plane(mesh, TAN, COORD))
    ind = np.arange(1, np.shape(COUPURE)[0], 2)
    COUPURE = np.delete(COUPURE, ind, axis = 0)

    return COUPURE

def ACP(COORD, CENTRE):
    """Effectue une ACP afin de définir l'axe expliquant le mieux la dispersion
    des données."""

    K = np.vstack((COORD,CENTRE))
    pca = PCA(n_components=2)
    pca.fit(K)
    B = pca.fit_transform(K)

    return B, pca

def Dist(COORD, Centre):
    """Calcul la distance des points au centre"""
    return distance.cdist(COORD, Centre)

def Densite(TRANSFORM, n, k):
    """Calcul la fonction de "densité" sur les axes k = 0 ou 1"""

    Interval = np.linspace(np.min(TRANSFORM[:,k]), np.max(TRANSFORM[:,k]), n)
    Count = np.zeros(np.size(Interval))
    for i in range(np.size(Interval)):
        Count[i] = np.count_nonzero(TRANSFORM[:,k]<=Interval[i])

    return Count, Interval

def Derivative(Count, Interval):
    """Calcul de la dérivé de la densité"""

    return np.gradient(Count, Interval)

def Find_nonvoid(Der):
    """ Extrait les endroits vides de la coupure en regardant là où la dérivée
    s'annule """

    Zero = np.where(Der <= 1.e-5)
    K = np.insert(Zero[0][1:]-Zero[0][:-1], 0, -1)
    Var_1 = 0
    Var_2 = 0
    liste = []
    for i in range(np.size(K)):

        if K[i] != 1:
            if Var_2 - Var_1 > 1:
                liste.append(Var_1)
                liste.append(Var_2)
            Var_1 = i
        elif K[i] == 1:
            Var_2 = i
        if i == np.size(K)-1 and Var_2 - Var_1 > 1:
            liste.append(Var_1)
            liste.append(Var_2)

    Non_void = Zero[0][liste]
    Non_void = np.insert(Non_void, 0, 0)
    Non_void = np.insert(Non_void, np.size(Non_void), np.size(Der)-1)
    Non_void = Non_void.reshape(int(np.size(Non_void)/2), 2)

    return Non_void

def Find_circle(Non_void, COORD, Center, Interval, k):
    """Permet de définir l'intervalle sans trou centré au point de la Spline
    considéré en calculant la distance minimum de chaque intervalle au point de la Spline"""

    Circle = []
    Dist_min = []

    for i in range(np.shape(Non_void)[0]):
        Circle.append(COORD[np.logical_and(COORD[:,k]<=Interval[Non_void[i,1]], COORD[:,k]>=Interval[Non_void[i,0]])])

    for i in range(len(Circle)):
        Dist_min.append(np.min(Dist(Circle[i][:,0:2], Center)))

    return Circle[np.argmin(Dist_min)]

def Isolation_Cercle(COORD, CENTRE):
    """On effectue plusieurs itérations de la méthode afin d'éliminer tout les cercles extérieurs. On se replace dans le plan en
    effectuant la transformation inverse de l'ACP"""

    n = 2
    while n > 1:
        TRANSFORM, pca = ACP(COORD, CENTRE)
        TRANSFORM_2 = TRANSFORM[0:-2,:]
        C = TRANSFORM[-1,:][np.newaxis]
        DIST = Dist(TRANSFORM_2, C)
        TRANSFORM_3 = TRANSFORM_2[np.hstack(DIST < 4*np.min(DIST))]
        Count, Interval = Densite(TRANSFORM_3[:,0:2], int(np.shape(TRANSFORM_3)[0]/5), 0)
        Count_2, Interval_2 = Densite(TRANSFORM_3[:,0:2], int(np.shape(TRANSFORM_3)[0]/5), 1)
        Der = Derivative(Count, Interval)
        Der_2 = Derivative(Count_2, Interval_2)
        Non_void = Find_nonvoid(Der)
        Non_void_2 = Find_nonvoid(Der_2)
        Circle = Find_circle(Non_void, TRANSFORM_3, C, Interval, 0)
        Circle_2 = Find_circle(Non_void_2, TRANSFORM_3, C, Interval_2, 1)
        if np.shape(Circle) < np.shape(Circle_2):
            Circle = pca.inverse_transform(Circle)
        else :
            Circle = pca.inverse_transform(Circle_2)
        COORD = Circle

        n = np.max((len(Non_void), len(Non_void_2)))

    return Circle

##################################################################################################################
###################### FONCTIONS POUR LE CHANGEMENT DE PLAN ET ETUDE DU R-THETA DANS LE PLAN #####################
##################################################################################################################

def Matrice_de_passage(TAN, NOR, BI):
    """Ressort la matrice de passage de la base canonique vers (NOR, BI, TAN). Le repère
    (NOR, BI, TAN) étant orthonormée l'inverse de la matrice de passage est sa transposée."""
    MAT_PASSAGE = np.zeros((3,3))
    MAT_PASSAGE[:,0] = NOR
    MAT_PASSAGE[:,1] = BI
    MAT_PASSAGE[:,2] = TAN

    return MAT_PASSAGE

def Changement_de_plan(PASSAGE, COORD, COUPURE):
    """ Effectue le changement de plan entre le repère initial et celui composé
    des vecteurs N,B,T. Retourne les coordonnées de la normale, binormale, du point
    central et de la coupure dans la nouvelle base. Ne retourne que les 2 premières
    coordonnées (celle par la tangente est invariente étant donné que l'on a découpé
    dans le plan orthogonal à la tangente)."""

    COORD_PLAN = ((np.dot(PASSAGE.T, COORD.T)).T)
    COUPURE_PLAN = ((np.dot(PASSAGE.T, COUPURE.T)).T)

    N_PLAN = np.asarray([[1, 0, 0]])
    B_PLAN = np.asarray([[0, 1, 0]])

    if round(np.max(COUPURE_PLAN[:,2]),3) - round(np.min(COUPURE_PLAN[:,2]),3) != 0 :
        print("ERREUR LA MATRICE DE PASSAGE EST FAUSSE")

    else :
        return COORD_PLAN, COUPURE_PLAN

def Theta_R_Experimental(COORD_PLAN, COUPURE_PLAN):
    """Extrait, pour chaque point de coupure, dans la nouvelle base, la distance de
    ce point au point de référence ainsi que l'angle entre ce point et la normale (coordonnée X
    dans notre nouvelle base). Retourne un tableau dont la première colonne est Theta et la
    seconde R (la distance). Le tableau est ordonné suivant l'axe Theta."""

    #CALCUL DU VECTEUR POINT - CENTRE
    VECT = COUPURE_PLAN - COORD_PLAN

    #LONGUEUR DU VECT = DISTANCE ENTRE LES 2 POINTS
    R = np.linalg.norm(VECT, axis = 1)

    #CALCUL LE THETA VIA LE PROD_SCAL AVEC LA NORMALE (JUSTE PREMIERE COORD DE VECT_NORM)
    #ET DONNE LA BONNE VALEUR ENTRE [0,2pi] EN FONCTION DU PROD_SCAL AVEC BINORMALE
    #(JUSTE LA SECONDE COLONNE DE VECT_NORM)
    VECT_NORM = VECT/R[np.newaxis].T
    Theta = np.arccos(VECT_NORM[:,0])
    DIR = VECT_NORM[:,1]

    Theta[np.where(DIR < 0)] = 2*pi - Theta[np.where(DIR < 0)]

    #ASSEMBLE R ET THETA DANS UN TABLEAU ET L'ORDONNE SUIVANT THETA
    TAB = np.hstack((Theta[np.newaxis].T, R[np.newaxis].T))
    TAB = TAB[np.argsort(TAB[:, 0])]

    return TAB

def Fourier_series(x, f, n = 0):
    """Construit la série de Fourier comme modèle de référence pour le fitting."""

    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x) for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))

    return series

def Modele_Fourier(THETA_R_EXP, ordre = 5):

    #INITIALISATION DU MODELE
    x, y = variables('x, y')
    model_dict = {y: Fourier_series(x, f = 1, n = ordre)}

    #SELECTION DES DONNEES POUR FITTING
    xdata = THETA_R_EXP[:,0]
    ydata = THETA_R_EXP[:,1]

    #FITTING VIA LA METHODE DE SYMPI
    fit = Fit(model_dict, x = xdata, y = ydata)
    fit_result = fit.execute()
    coeffs = fit_result.params
    COEFFS = np.asarray(list(coeffs.values()))[np.newaxis].T
    erreur = fit_result.r_squared

    return COEFFS, erreur

def Compute_Serie_Fourier(theta, a0, coeff_a, coeff_b):

    serie = np.zeros(len(theta))
    for j in range(len(theta)) :
        sum = a0
        for i in range(len(coeff_a)):
            sum += coeff_a[i]*np.cos((i+1)*theta[j]) + coeff_b[i]*np.sin((i+1)*theta[j])
        serie[j] = sum
    return serie

##################################################################################################################
################################ FONCTIONS POUR INTERPOLATION ENTRE DEUX BASES ###################################
##################################################################################################################

def find_interval(liste_t, target):

    liste_t.append(target)
    new = sorted(liste_t)
    indices = new.index(target)
    a = indices - 1
    b = indices

    return a, b

def Solution_geodesique(PHI, PSY, t):

    U, SIGMA, V_T = Svd(PHI, PSY)

    V = np.transpose(V_T)
    N = np.shape(SIGMA)[0]

    COS_T_SIGMA = np.zeros((N,N))
    SIN_T_SIGMA = np.zeros((N,N))

    for i in range(N):
        COS_T_SIGMA[i, i] = np.cos(t*SIGMA[i,i])
        SIN_T_SIGMA[i, i] = np.sin(t*SIGMA[i,i])

    #SOLUTION DE LA FORME : GAMMA(t) = GAMMA(0).V.cos(SIGMA*t) + U.sin(SIGMA*t)
    P1 = np.dot(np.dot (PHI,V), COS_T_SIGMA)
    P2 = np.dot(U, SIN_T_SIGMA)

    GAMMA_t = P1 + P2

    return GAMMA_t

def Svd(PHI, PSY):

    PHI_PHI_T = np.dot(PHI, np.transpose(PHI))
    N = np.shape(PHI_PHI_T)[0]
    I = np.eye(N)
    PHI_T_PSY = np.dot(np.transpose(PHI) , PSY)
    INV_PHI_T_PSY = np.linalg.inv(PHI_T_PSY)

    #INITIALISATION DE LA MATRICE SUR LEQUEL ON VA FAIRE LA SVD
    POUR_SVD = np.dot( np.dot( I - PHI_PHI_T , PSY ), INV_PHI_T_PSY )

    #SVD (ATTENTION : LE SIGMA SORTANT DE np.linalg.svd EST UN VECTEUR (k,))
    U, tnSIGMA, V_T = np.linalg.svd(POUR_SVD, full_matrices = False)

    SIGMA = np.arctan(tnSIGMA)

    #LE VECTEUR SIGMA TRANSFORME EN MATRICE (k,k)
    SIGMA = np.diag(SIGMA)

    return U, SIGMA, V_T

##################################################################################################################
################################ FONCTION POUR LA RECONSTRUCTION DES POINTS APPROXIMES ###########################
##################################################################################################################

def Reconstruction_contour(COORD_PLAN, TAB, PASSAGE):

    liste_point = []
    for i in range(np.shape(TAB)[0]):
        theta = TAB[i,0]
        r = TAB[i, 1]
        rotation = np.asarray([np.cos(theta), np.sin(theta), 0])
        vect = r*rotation
        new_point_plan = COORD_PLAN + vect
        new_point_canon = np.dot(PASSAGE, new_point_plan.T).T
        liste_point.append(new_point_canon)

    return np.vstack(liste_point)
