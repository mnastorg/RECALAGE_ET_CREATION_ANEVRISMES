import os
import time

import numpy as np
import trimesh

import BSplines_Utilities as bs
import Gestion_Fichiers as gf
import Creation_Base_Main as crea_basis

gf.Reload(gf)
gf.Reload(bs)
gf.Reload(crea_basis)

def Main_Espace(path_centerline, path_mesh, liste_t):

    start_time = time.time()
    liste_centerline = sorted(os.listdir(path_centerline))
    liste_stl = sorted(os.listdir(path_mesh))

    liste_control = []
    liste_mesh = []
    for i in range(len(liste_centerline)):
        CONTROL = gf.Lecture_csv_pcl(path_centerline + liste_centerline[i])
        liste_control.append(CONTROL)
        mesh = trimesh.load_mesh(path_mesh + liste_stl[i])
        liste_mesh.append(mesh)
    lecture_time = time.time()
    print("Temps d'importation des données : ", round(lecture_time - start_time, 3))

    ######## PARAMETRES ######################
    nb_points_reconstruction = 200
    degree = 3
    ordre_fourier = 5
    nb_vect_base = len(liste_centerline)
    ##########################################

    print("-------------------------- BASE CENTERLINE -----------------------------")
    BASE_CENTERLINE = crea_basis.Main_base_centerline(liste_control, nb_points_reconstruction, degree, nb_vect_base)
    center_time = time.time()
    print("Temps création base centerline : ", round(center_time-lecture_time, 3))
    print("-------------------------- BASE CONTOUR --------------------------------")
    liste_base = crea_basis.Main_base_contour(liste_control, liste_mesh, liste_t, nb_vect_base, degree, ordre_fourier)
    end_time = time.time()
    print("Temps création base contour : ", round(end_time - center_time, 3))
    print("Temps  total méthode : ", round(end_time - start_time, 3))

    return BASE_CENTERLINE, liste_base
