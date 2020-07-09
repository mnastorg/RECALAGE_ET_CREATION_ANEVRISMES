import numpy as np
import trimesh
from scipy.spatial import distance
import Gestion_Fichiers as gf
import time
from sklearn.neighbors import NearestNeighbors


liste_p001 = [0, 2, 6, 19, 25]
mesh_p001 = trimesh.load_mesh("p001.stl")
RECONSTRUCTION_p001 = gf.Lecture_csv_bsplines("RESULTATS/RECONSTRUCTION.csv")

def Intersection_plane(nbr_voisins, mesh = mesh_p001, RECONSTRUCTION = RECONSTRUCTION_p001, liste = liste_p001):

    start_time = time.time()
    Principal = []
    for i in liste:
        Principal.append(RECONSTRUCTION[RECONSTRUCTION[:,3] == i])

    PRINCIPAL = np.vstack(Principal)

    gf.Write_csv("BranchePrincipal.csv", PRINCIPAL[:,0:3], "x , y, z")

    Intersec = []
    for i in range(np.shape(PRINCIPAL)[0]):
        A = np.vstack(trimesh.intersections.mesh_plane(mesh, PRINCIPAL[i, 6:9], PRINCIPAL[i, 0:3]))

        #Problème : Cette intersection ne s'arrète pas seulement aux parois de la branche principale, il faut donc trouver un critère pour ne conserver que les points qui nous intéressent (Ici, nous utilisons le label des k voisins du squelette initial )
        neigh = NearestNeighbors(n_neighbors = nbr_voisins)
        neigh.fit(RECONSTRUCTION[:,0:3])
        B = neigh.kneighbors(A, nbr_voisins, return_distance=False)

        LABS = RECONSTRUCTION[B,3].astype(int)

        LABS_freq = np.zeros((np.shape(LABS)[0], 1))
        for j in range(np.shape(LABS)[0]):
            L = list(set(LABS[j,:]))
            K = []
            for k in L:
                K.append(np.count_nonzero(LABS[j,:] == k))

            LABS_freq[j,:] = L[np.argmax(K)]

        delete = []
        for j in range(np.size(LABS_freq)):
            if LABS_freq[j] not in liste_p001:
                delete.append(j)


        A = np.delete(A, delete, axis = 0)
        Intersec.append(A)

    Intersec = np.vstack(Intersec)
    gf.Write_csv("Maybe.csv", Intersec, "x, y, z")
    end_time = time.time()
    print("Temps total de la méthode : ", round(end_time - start_time, 2))
    return Intersec
