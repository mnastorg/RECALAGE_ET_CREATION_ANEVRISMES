import numpy as np
from math import *

from scipy.spatial import distance

import Gestion_Fichiers as gf
import Statistiques_Utilities as stats
import BSplines_Utilities as bs

gf.Reload(stats)
gf.Reload(bs)

#########################################################################################################################################
###################################### FONCTION POUR LA RECONSTRUCTION DU SQUELETTE DE NUREA ############################################
#########################################################################################################################################

def Reconstruction(NUREA):
    """Effectue une reconstruction par branche en Bsplines du squelette de Nurea.
    Ressort un tableau
    x, y, z, label, is_sail, is_leaf, tan x, tan y, tan z, nor x, nor y, nor z, bi x, bi y, bi z, courbure"""

    #SELECTION DES LABELS
    label = list(set(NUREA[:,3]))

    tab = []

    #ON PARCOURT CHAQUE BRANCHE
    for i in label :
        BRANCHE = NUREA[NUREA[:,3] == i]
        nb_points = np.shape(BRANCHE)[0]
        KNOT = bs.Knotvector(BRANCHE[:,0:3], 4)
        BSPLINES, D1, D2, D3 = bs.BSplines_RoutinePython(BRANCHE[:,0:3], KNOT, 4, np.linspace(0, 1, nb_points), dimension = 3)
        #ON DECALE TOUS LES INDICES DE LABELS DE -1
        BSPLINES = np.insert(BSPLINES, 3, BRANCHE[:,3] - 1, axis = 1)
        BSPLINES = np.insert(BSPLINES, 4, BRANCHE[:,4], axis = 1)
        BSPLINES = np.insert(BSPLINES, 5, BRANCHE[:,5], axis = 1)

        #CALCUL TANGENTE, NORMALE ET BINORMALE
        TAN = stats.tangente(D1)
        BI = stats.binormale(D1, D2)
        NOR = stats.normale(BI, TAN)

        #CALCUL DE LA COURBURE
        courbure = stats.courbure(D1, D2)

        #CALCUL DE LA TAILLE DE L'ARC
        longueur = stats.liste_length_point(np.linspace(0, 1, nb_points), BRANCHE[:,0:3], 4, KNOT)
        longueur = np.asarray(longueur)[np.newaxis].T

        TAB = np.hstack((BSPLINES, TAN, NOR, BI, courbure, longueur))

        tab.append(TAB)

    SKELET = np.vstack(tab)

    return SKELET

#########################################################################################################################################
###################################### FONCTION POUR L'INITIALISATION DES POINTS DE CONTROLE ############################################
#########################################################################################################################################

def Nb_points_label(RECONSTRUCTION):
    """Fonction permettant de sortir le nombre de points, pour chaque label dans le squelette de NUREA
    ainsi que le nombre total de points dans le squelette."""

    label = list(set(RECONSTRUCTION[:,3]))
    nb_pts = []

    for k in label :
        nb_pts_label = np.shape(RECONSTRUCTION[RECONSTRUCTION[:,3]==k])[0]
        nb_pts.append(nb_pts_label)

    somme = np.sum(nb_pts)

    return nb_pts, somme

def Change_intervalle(x, old_min, old_max, new_min, new_max):
    """Convertit un nombre dans l'intervalle [old_min, old_max] en
    un nombre dans l'intervalle [new_min, new_max]"""

    a = (new_max - new_min)/(old_max - old_min)
    b = new_min - a*old_min

    return a*x + b

def Nb_points_controle_label(RECONSTRUCTION):
    """ Fonction ressortant le nombre de points de controle en fonction
    de la courbure et de la taille de l'arc, pour chaque branche. """

    labels = list(set(RECONSTRUCTION[:,3]))

    mean_curve = []
    max_length = []
    value = []

    for i in labels :
        BRANCHE = RECONSTRUCTION[RECONSTRUCTION[:,3] == i]
        courbure = BRANCHE[:,15]
        longueur = BRANCHE[:,16]
        mean_curve.append(np.mean(courbure))
        max_length.append(np.max(longueur))
        value.append(np.mean(courbure)*np.max(longueur))

    nb_controle = []
    for i in range(len(value)) :
        nb_controle.append(floor(Change_intervalle(value[i], min(value), max(value), 3, 10)))

    return nb_controle

def Creation_controles(RECONSTRUCTION):
    """ Fonction sortant les points de contrôles initiaux. Pour chaque intersection où
    un seul point saillant est mentionné sur une seule des branches, on l'ajoute dans les
    autres branches de manière à ce que le squelette initial soit parfaitement relié."""

    label = list(set(RECONSTRUCTION[:,3]))
    nb_pts_label, somme = Nb_points_label(RECONSTRUCTION)
    #ON DECALE DE -1 POUR QUE LE LINSPACE NE CREE PAS D'ERREUR
    #EN PRENANT UN INDICE EXTREME TROP GRANDE
    nb_pts_label = [i - 1 for i in nb_pts_label]
    nb_pts_controle = Nb_points_controle_label(RECONSTRUCTION)

    pt_ex = []
    pt_inter = []

    #ON PARCOURT LES LABELS, NB DE POINTS ET NB POINTS DE CONTROLES
    #LE BUT DE CETTE BOUCLE EST D'ISOLER LES POINTS EXTREMES QUI
    #SONT SAILLANTS OU NON ET D'ISOLER LES POINTS INTERNES
    for k, i, j in zip(label, nb_pts_label, nb_pts_controle):

        LAB = RECONSTRUCTION[RECONSTRUCTION[:,3] == k]
        #RECONSTRUCTION EST ORDONNEE LES PT 1 ET 2 SONT DONC
        #LES EXTREMITES DE LA BRANCHE
        pt1 = LAB[0,:]
        pt2 = LAB[-1,:]
        pt = []
        pt.append(pt1)
        pt.append(pt2)
        pt_ex.append(pt)

        #ON SELECTIONNE LE NOMBRE DE POINTS DE CONTROLE VOULUS
        #SUR LE NOMBRE TOTAL DE POINTS DANS LA BRANCHE PUIS ON RETIRE
        #LES DEUX EXTREMITES
        T = np.linspace(0, i, j, dtype = 'int')
        INTERN = LAB[T,:]
        INTERN = INTERN[1:-1,:]

        pt_inter.append(INTERN)

    #A CE STADE NOUS AVONS DEUX LISTES UNE POUR LES POINTS EXTREMES
    #ET L'AUTRE POUR LES POINTS INTERNES

    #DANS CETTE PARTIE ON S'INTERESSE A ASSOCIER, POUR TOUT POINT EXTREME
    #NON SAILLANT ET NON LEAF SON POINT SAILLANT LE PLUS PROCHE

    MAT = np.vstack(pt_ex)
    SAILL = MAT[MAT[:,4] == 1]
    PAS_SAILL = MAT[np.logical_and(MAT[:,4] == 0, MAT[:,5] == 0)]
    DIST = distance.cdist(SAILL[:,0:3], PAS_SAILL[:,0:3])
    Cor = np.argmin(DIST, axis = 0)
    COR = SAILL[Cor]

    #DANS CETTE BOUCLE ON REMPLACE LES POINTS NON SAILLANTS ET NON LEAF PAR LEUR
    #EQUIVALENT SAILLANT

    for i, j in zip(MAT[np.logical_and(MAT[:,4] == 0, MAT[:,5] == 0)], COR):
        MAT[np.all(MAT==i, axis = 1)] = j

    #FINALEMENT IL NE NOUS RESTE PLUS QU'A RELIER LES NOUVEAUX POINTS EXTREMES
    #AVEC LEUR POINTS INTERNES EQUIVALENTS

    k = int (np.shape(MAT)[0]/2)
    control = []
    for i, j in zip(range(k), pt_inter):
        j = np.insert(j, 0, MAT[2*i,:], axis = 0)
        j = np.insert(j, np.shape(j)[0], MAT[2*i+1,:], axis = 0)
        #ON MET LES BONS LABELS POUR LES POINTS EXTREMES
        j[0,3] = j[1,3]
        j[-1,3] = j[1,3]
        control.append(j[:,0:6])

    return control

#########################################################################################################################################
###################################### FONCTION POUR L'INITIALISATION DES BSPLINES ######################################################
#########################################################################################################################################

def Liste_pour_linspace(RECONSTRUCTION):
    """ Fonction ressortant une liste de linspace pour reconstruire
    les BSplines avec le même nombre de points que le squelette de Nurea"""

    nb_points, somme = Nb_points_label(RECONSTRUCTION)
    liste_t = []

    for i in nb_points :
        liste_t.append(np.linspace(0,1,i))

    if Nb_points_liste(liste_t)[1] == somme :
        return liste_t

    else:
        print("Erreur pas le même nombre de points que le squelette initial")

    return liste_t

def Nb_points_liste(liste):
    """ Fonctions ressortant le nombre de points d'une liste de tableaux """
    l = []

    for i in range(len(liste)):
        l.append(len(liste[i]))

    somme = np.sum(l)

    return l, somme

def Ajout_saillants_leafs_bsplines(BSPLINES, CONTROL):
    """Fonction permettant d'ajouter dans le tableau des bsplines les points saillants et leafs
    correspondants """
    BSPLINES = np.insert(BSPLINES, 4, 0, axis = 1)
    BSPLINES = np.insert(BSPLINES, 5, 0, axis = 1)

    SAILLANT = CONTROL[CONTROL[:,4] == 1]
    LEAF = CONTROL[CONTROL[:,5] == 1]

    for i in range(np.shape(SAILLANT)[0]):
        index_sail = np.where(np.all(BSPLINES[:,0:3] == SAILLANT[i,0:3], axis = 1))[0]
        BSPLINES[index_sail, 4] = 1

    if np.shape(LEAF)[0] > 0 :
        for i in range(np.shape(LEAF)[0]):
            index_leaf = np.where(np.all(BSPLINES[:,0:3] == LEAF[i,0:3], axis = 1))[0]
            BSPLINES[index_leaf, 5] = 1

    return BSPLINES

def Creation_bsplines(control_init, degree, liste_t):
    """Fonction qui retourne tous les paramère nécessaires pour la BSplines à partir
    des points de controles initiaux. En particulier elle retourne : la liste des noeuds
    la matrice de base d'ordre 0, 1 et 2, la liste des bsplines avec leurs labels, la mention
    saillant et leaf ainsi que le repere de Frenet, la liste des dérivées premieres et seocondes"""

    #INITIALISATION DES LISTES
    liste_knot = []
    liste_basis = []
    liste_basis_der1 = []
    liste_basis_der2 = []
    liste_bsplines = []
    liste_der1 = []
    liste_der2 = []

    #CALCUL DES BSPLINES POUR CHAQUE BRANCHE
    for i in range(len(control_init)):
        CONTROL = control_init[i]

        knot = bs.Knotvector(CONTROL[:,0:3], degree)
        BASIS = bs.Matrix_Basis_Function(degree, knot, np.shape(CONTROL)[0], liste_t[i], 0)
        BASIS_DER1 = bs.Matrix_Basis_Function(degree, knot, np.shape(CONTROL)[0], liste_t[i], 1)
        BASIS_DER2 = bs.Matrix_Basis_Function(degree, knot, np.shape(CONTROL)[0], liste_t[i], 2)

        #CALCUL DES BSPLINES ET AJOUT DES LABELS
        BSPLINES = np.dot(BASIS, CONTROL[:,0:3])
        BSPLINES = np.insert(BSPLINES, 3, CONTROL[0,3], axis = 1)
        BSPLINES = Ajout_saillants_leafs_bsplines(BSPLINES, CONTROL)

        #CALCUL DES DERIVEES 1 ET 2
        DER1 = np.dot(BASIS_DER1, CONTROL[:,0:3])
        DER2 = np.dot(BASIS_DER2, CONTROL[:,0:3])

        #CALCUL DU REPERE DE FRENET POUR CHAQUE POINT
        TAN = stats.tangente(DER1)
        BI = stats.binormale(DER1, DER2)
        NOR = stats.normale(BI, TAN)

        BSPLINES = np.hstack((BSPLINES, TAN, NOR, BI))

        liste_knot.append(knot)
        liste_basis.append(BASIS)
        liste_basis_der1.append(BASIS_DER1)
        liste_basis_der2.append(BASIS_DER2)
        liste_bsplines.append(BSPLINES)
        liste_der1.append(DER1)
        liste_der2.append(DER2)

    return liste_knot, liste_basis, liste_basis_der1, liste_basis_der2, liste_bsplines, liste_der1, liste_der2

#########################################################################################################################################
###################################### FONCTION POUR DETERMINER LES CONNECTIVITES #######################################################
#########################################################################################################################################

def Connectivite(control_init, liste_bsplines):
    """Fonction permetttant d'identifier les connectivités. Dans control_init tous les points d'une
    bifurcation sont saillants. Ici on se donne pour but de déterminer les points, dans une bifurcation,
    qui connectent les splines. Les points restants restent saillants et pourront bouger librement.
    Le fonction ressort les points de controles avec leurs nouvelles caracteristique saillants
    ainsi qu'une liste des points a merge."""

    #EXTRACTION DES POINTS SAILLANTS VIA LES POINTS DE CONTROLES
    CONTROL = np.vstack(control_init)
    index_sail = np.where(CONTROL[:,4] == 1)
    SAILLANT = CONTROL[index_sail]

    #INITIALISATION DU TABLEAU RELATIF A liste_bsplines
    BSPLINES = np.vstack(liste_bsplines)

    #ON CAPTURE LES POINTS SAILLANTS D'UNE INTERSECTION (i.e CEUX QUI SONT
    #AU MEME ENDROIT DONC DE DISTANCE 0)
    groupe_saillants = []
    DIST = distance.cdist(SAILLANT[:,0:3], SAILLANT[:,0:3])
    for i in range(np.shape(DIST)[0]):
        groupe_saillants.append(np.where(DIST[i,:] == 0)[0])

    #RETOURNE UN GROUPE D'INDICES DE SAILLANTS QUI SE
    #REJOIGNENT AU MEME ENDROIT EN EVITANT LES DOUBLONS DE
    #groupe_saillants. IL Y A AUTANT DE BIFURCATIONS QUE D'
    #ELEMENTS DANS INTER_INIT
    inter_init = list(set(map(tuple, groupe_saillants)))

    #ON EXTRAIT DES TRIPLETS QUI NOUS DONNE L'INFORMATION, POUR CHAQUE
    #BIFURCATION DES BRANCHES QUI SE REJOIGNENT
    label_init = []
    for k in inter_init:
        label_init.append(CONTROL[index_sail[0][list(k)],3].astype(int))

    #POUR TOUTE BIFURCATION ON ASSOCIE LE POINT EQUIVALEMENT DANS liste_bsplines
    #DE LA FORME [X, 0, 0] SIGNIFIE QUE LE POINT EST LE Xeme DANS UN LABEL (LE DERNIER)
    #ET LE PREMIER POINT POUR LES AUTRES LABELS.
    pts_bsplines = []
    for i, j in zip(label_init, inter_init):
        pts = []
        for k in i:
            DIST = distance.cdist(CONTROL[index_sail[0][j[0]],0:3][np.newaxis], liste_bsplines[k][:,0:3])
            pts.append(np.argmin(DIST))
        pts_bsplines.append(pts)

    #ICI, ON ASSOCIE POUR TOUTE BIFURCATION LES VECTEURS TANGENTS CORRESPONDANTS
    tangente = []
    for i, j in zip(label_init, pts_bsplines):
        vect = []
        for k, l in zip(i, j):
            if (liste_bsplines[k][l,6:9] != 0).any() == True:
                vect.append(liste_bsplines[k][l,6:9])
            elif np.shape(liste_bsplines[k])[0] == l+1:
                vect.append(liste_bsplines[k][l-1,6:9])
            else:
                vect.append(liste_bsplines[k][l+1,6:9])

        tangente.append(vect)

    #ON FAIT LE PRODUIT SCALAIRE ENTRE LES VECTEURS TANGENTS
    #LES DEUX VECTEURS DONC LA VALEUR DU PRODUIT SCALAIRE EST
    #MAXIMALE SONT DITS "CONNECTES"
    connecte = []
    for i in tangente :
        V = np.vstack(i)
        PDT_V = np.matmul(V, V.T)
        PDT_V = np.triu(PDT_V, 1)
        PDT_V[PDT_V == 0]= -50
        #ind = np.unravel_index(np.argmax(PDT_V, axis = None), PDT_V.shape)
        ind_2 = np.vstack(np.where(np.abs(PDT_V - np.max(PDT_V)) <= 0.1))
        ind_2 = tuple(list(set(ind_2.flatten())))
        connecte.append(ind_2)

    #ON CREE LE TABLEAU QUI RETOURNE LES POINTS A "MERGE" PLUS TARD
    merge = []
    for i,j in zip(inter_init, connecte):
        id = []
        for k in j:
            id.append(index_sail[0][i[k]])
        merge.append(id)

    #FINALEMENT ON APPLIQUE AUX POINTS CONNECTE LA VALEUR DE 0
    #LAISSANT UNIQUEMENT LES POINTS SAILLANTS QUE L'ON LAISSERA
    #LIBRE DE SE DEPLACER PAR LA SUITE
    NEW_CONTROL = np.copy(CONTROL)
    for i in merge:
        if len(i) == 3:
            NEW_CONTROL[i[1:],4] = 0
        else:
            NEW_CONTROL[i, 4] = 0
    label = list(set(BSPLINES[:,3].astype(int)))
    liste_control = []
    for i in label:
        liste_control.append(NEW_CONTROL[NEW_CONTROL[:,3] == i])


    return liste_control, merge
