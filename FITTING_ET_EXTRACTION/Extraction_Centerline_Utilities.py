import numpy as np

from sklearn.neighbors import NearestNeighbors

import Gestion_Fichiers as gf
import BSplines_Utilities as bs
import Statistiques_Utilities as stats

def Statistiques_sur_bsplines(new_bsplines, new_der1, new_der2, LEVELSET):

    tab = []

    for i in range(len(new_bsplines)) :
        BRANCHE = new_bsplines[i]
        DER1 = new_der1[i]
        DER2 = new_der2[i]

        TAN = stats.tangente(DER1)
        BI = stats.binormale(DER1, DER2)
        NOR = stats.normale(BI, TAN)

        courbure = stats.courbure(DER1, DER2)

        TAB = np.hstack((BRANCHE, TAN, BI, NOR, courbure))
        tab.append(TAB)

    TAB = np.vstack(tab)

    rayon = stats.rayon_vaisseau(TAB, LEVELSET)

    TAB = np.hstack((TAB, rayon))

    return TAB

def Moyenne_rayon(TAB):

    moy = []
    labels = list(set(TAB[:,3]))

    for i in labels :
        LAB = TAB[TAB[:,3] == i]
        rayon = LAB[:,16]
        moy.append(np.mean(rayon))

    return moy

def Point_de_depart(TAB):

    #ON SORT LA MOYENNE DES RAYONS
    moy = Moyenne_rayon(TAB)

    #ON CHERCHE DANS LES SPLINES QUI ONT UN LEAF
    #CELLE QUI A LA PLUS GRANDE MOYENNE DE RAYON
    LEAF = TAB[TAB[:,5] == 1]
    lab_leaf = list(set(LEAF[:,3]))
    leaf_moy_rayon = [moy[int(i)] for i in lab_leaf]
    first_lab = np.argmax(leaf_moy_rayon)

    #ON SELECTIONNE NOTRE PREMIER POINT LEAF DE DEPART
    LAB_DEPART = TAB[TAB[:,3] == first_lab]
    pt_depart = LAB_DEPART[LAB_DEPART[:,5] == 1]

    return pt_depart

def Labels_bsplines_principale(TAB):

    liste_labels = []

    moyenne = Moyenne_rayon(TAB)
    MAX = np.max(moyenne)

    current_point = Point_de_depart(TAB)
    current_lab = int(current_point[:,3])
    liste_labels.append(current_lab)

    arret = False
    it = 0

    while arret == False and it < len(moyenne) :

        CURRENT = TAB[TAB[:,3] == current_lab]

        if np.all(current_point == CURRENT[0,:]) :
            end_point = CURRENT[-1,:][np.newaxis]
        else :
            end_point = CURRENT[0,:][np.newaxis]

        nbrs = NearestNeighbors(n_neighbors = 5, algorithm='auto').fit(TAB[:,0:3])
        distances, indices = nbrs.kneighbors(end_point[:,0:3])
        sail = (TAB[indices[0]])[:,4]

        if np.any(sail == 1) == False :

            SUPPR_TAB = np.delete(TAB, np.where(TAB[:,3] == current_lab), axis = 0)
            nbrs = NearestNeighbors(n_neighbors = 1, algorithm='auto').fit(SUPPR_TAB[:,0:3])
            distances, indices = nbrs.kneighbors(end_point[:,0:3])

            current_point = SUPPR_TAB[indices[0]]
            current_lab = int(current_point[:,3])
            liste_labels.append(current_lab)

        else :
            labels = list(set((TAB[indices[0]])[:,3]))
            labels = [int(i) for i in labels]
            del labels[labels.index(current_lab)]
            moy = [moyenne[i] for i in labels]
            maxi = np.max(moy)

            if maxi < MAX/2 :
                arret = True

            else :
                argmaxi = np.argmax(moy)
                lab_branche = labels[argmaxi]

                BRANCHE = TAB[TAB[:,3] == lab_branche]
                nbrs = NearestNeighbors(n_neighbors = 1, algorithm='auto').fit(BRANCHE[:,0:3])
                distances, indices = nbrs.kneighbors(end_point[:,0:3])

                current_point = BRANCHE[indices[0]]
                current_lab = int(current_point[:,3])
                liste_labels.append(current_lab)

        it += 1

    return liste_labels

def Branche_principale(TAB):

    labels = Labels_bsplines_principale(TAB)
    principale = []

    for i in labels :
        principale.append(TAB[TAB[:,3] == i])

    PRINCIPALE = np.vstack(principale)

    return PRINCIPALE, labels

def Creation_bsplines_principale(PRINCIPALE, nb_control = 10, degree = 5):

    nb_points = np.shape(PRINCIPALE)[0]
    T = np.linspace(0, nb_points-1, nb_control, dtype = 'int')
    t = np.linspace(0, 1, nb_points)

    CONTROL = PRINCIPALE[T,:]
    KNOT = bs.Knotvector(CONTROL, degree)

    BASIS = bs.Matrix_Basis_Function(degree, KNOT, np.shape(CONTROL)[0], t, 0)
    BASIS_DER1 = bs.Matrix_Basis_Function(degree, KNOT, np.shape(CONTROL)[0], t, 1)
    BASIS_DER2 = bs.Matrix_Basis_Function(degree, KNOT, np.shape(CONTROL)[0], t, 2)

    #CALCUL DES BSPLINES ET AJOUT DES LABELS
    BSPLINES = np.dot(BASIS, CONTROL)

    return CONTROL, BSPLINES, KNOT, BASIS, BASIS_DER1, BASIS_DER2, t
