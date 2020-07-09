import numpy as np
import trimesh

import time

import Nuage_Level_Set_Utilities as nlsu
import Gestion_Fichiers as gf

def Main_nuage_level_set(file_stl, pitch, seuil):

    gf.Reload(nlsu)

    start_time = time.time()

    aorte = trimesh.load_mesh(file_stl)
    print("Nombre de sommets du mesh = ", len(aorte.vertices))

    #TRAITEMENT STL INITIAL
    VOX,vox = nlsu.Voxelisation(aorte, pitch)

    print("Nombre de voxels provenant du maillage = ", vox.filled_count)

    FILL, fill = nlsu.Voxel_fill(vox)

    print("Nombre de voxels apres remplissage = ", fill.filled_count)
    print("Taille de la grille = ", fill.shape)

    BORDURE = nlsu.Voxel_bordure(VOX,FILL)

    #EXTRACTION DES INDICES RELATIFS AUX POINTS
    indices = nlsu.Indices_matrix(fill)
    indices_geo = nlsu.Indices_geo(fill, indices)

    step1 = time.time()
    print("Temps voxelisation = ", round(step1 - start_time, 2))

    #LEVELSET SANS LES POINTS EXTERIEURS A LA GEOMETRIE
    LEVELSET = nlsu.Level_set(BORDURE)
    INDICES_LEVELSET = nlsu.Ajout_indices(LEVELSET, indices_geo, FILL)
    #gf.Write_csv("LEVEL_SET.csv", INDICES_LEVELSET, "x, y, z, scalaire")

    step2 = time.time()
    print("Temps FastMarching = ", round(step2 - step1, 2))

    #NOYAU POUR LA REGULARISATION
    G3 = nlsu.Noyau_gaussien_3D(0.5,2)
    REG = nlsu.Convolution(LEVELSET, G3)
    INDICES_REG = nlsu.Ajout_indices(REG, indices_geo, FILL)

    step3 = time.time()
    print("Temps convolution gaussienne = ", round(step3 - step2, 2))

    #NOYAU POUR FILTRE LAPLACIEN
    L7 = nlsu.Noyau_laplacien_7()
    LAP = nlsu.Convolution(REG, L7)
    INDICES_LAP = nlsu.Ajout_indices(LAP, indices_geo, FILL)

    step4 = time.time()
    print("Temps convolution laplacien = ", round(step4 - step3, 2))

    #EXTRACTION DU NUAGE DE POINT PAR UN SEUIL PRECIS
    PCL = nlsu.Extraction_pcl(INDICES_LAP, seuil)
    #gf.Write_csv("PCL.csv", PCL, "x, y, z")

    step5 = time.time()
    print("Temps total extraction PCL = ", round(step5 - start_time, 2))

    return PCL, INDICES_LEVELSET
