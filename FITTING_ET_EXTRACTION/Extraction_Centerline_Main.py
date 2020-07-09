import numpy as np
import time

import Gestion_Fichiers as gf
import Extraction_Centerline_Utilities as extract
import BSplines_Utilities as bs

gf.Reload(gf)
gf.Reload(extract)
gf.Reload(bs)

def Main_Extraction_Centerline_Auto(new_bsplines, new_der1, new_der2, LEVELSET):

    start_time = time.time()

    #ON EFFECTUE TOUTES LES STATISTIQUES SUR NOTRE GRANDE GEOMETRIE APRES
    #SON FITTING
    TAB = extract.Statistiques_sur_bsplines(new_bsplines, new_der1, new_der2, LEVELSET)
    step1 = time.time()
    print("Temps pour étape 1 : Calcul des stats sur la centerline = ", round(step1 - start_time, 2))

    #ON EXTRAIT LA BRANCHE PRINCIPALE
    PRINCIPALE, lab_principale = extract.Branche_principale(TAB)
    PRINCIPALE = PRINCIPALE[:,0:3]
    branche_principale = [lab_principale, PRINCIPALE]
    step2 = time.time()
    print("Temps pour étape 2 : Extraction BSplines principale = ", round(step2 - step1, 2))

    #ON SORT LA NOUVELLE BSPLINES
    CONTROL, BSPLINES, KNOT, BASIS, BASIS_DER1, BASIS_DER2, t = extract.Creation_bsplines_principale(PRINCIPALE, nb_control = 5, degree = 3)
    centerline_parametres = [CONTROL, BSPLINES, KNOT, BASIS, BASIS_DER1, BASIS_DER2, t]
    step3 = time.time()
    print("Temps pour étape 3 : Création de la courbe centrale", round(step3 - step2, 2))

    return TAB, branche_principale, centerline_parametres

def Main_Extraction_Centerline_Labels(new_bsplines, liste_label, nb_control = 5, degree = 3):

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
