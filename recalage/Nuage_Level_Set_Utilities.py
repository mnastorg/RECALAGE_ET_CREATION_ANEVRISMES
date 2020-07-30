###################################################################################################################################
###################################### MODULE DES FONCTIONS POUR LE MAIN DU NUAGE DE POINT ########################################
###################################################################################################################################

import numpy as np
import trimesh
import skfmm
import scipy.signal

####################################################################################
############################### ELEMENTS POUR LA VOXELISATION ######################
####################################################################################

def Voxelisation(mesh, pitch):
    """Retourne VOX format np.array et vox format trimesh correspondant à la
    voxelisation de geom avec une précision pitch"""
    vox = mesh.voxelized(pitch)
    VOX = vox.matrix

    return VOX, vox

def Voxel_fill(vox):
    """Retourne FILL format np.array et fill format trimesh. Les matrices remplisse
    vox avec des voxel (on cherche à remplir la triangulation de surface initiale)"""
    fill = vox.fill()
    FILL = fill.matrix

    return FILL, fill

def Voxel_bordure(VOX, FILL):
    """Retourne la matrice GEOM_DIST à utiliser dans Distance.py pour appliquer
    la fonction de FastMarching python. 0 si frontière, 1 intérieur et -1 extérieur"""
    GEOM_DIST = 0*FILL
    GEOM_DIST[FILL == False] = -1
    GEOM_DIST[FILL != VOX] = 1

    return GEOM_DIST

def Indices_matrix(fill):
    """Retourne un tableau des coordonnées correspondant aux voxels TRUE (attention
    il faudra s'adapter pour que nos tableaux soient de la taille d'indices_geo)"""
    indices = fill.sparse_indices
    return  indices

def Indices_geo(fill,indices):
    """Retourne à partir des indices matriciels les indices de la geo originale"""
    indices_geo = fill.indices_to_points(indices)
    return indices_geo

####################################################################################
############################### ELEMENTS POUR LA LEVEL SET #########################
####################################################################################

def Level_set(BORDURE):
    """ Retourne la matrice DISTZERO de fonction distance relative à GEOM_DIST
    où les valeurs aux points extérieurs sont 0"""
    DIST = skfmm.distance(BORDURE, order = 1)
    DISTZERO = DIST.copy()
    DISTZERO[DISTZERO < 0] = 0

    return DISTZERO

def Ajout_indices(LEVELSET, indices_geo, FILL):
    """Retourne un tableau près à l'affichage de type x/y/z/scalaire avec scalaire
    issu de LEVELSET et x/y/z issus de indices_geo. FILL permet de supprimer les éléments
    extérieurs à la geo dans LEVELSET"""

    [n1, n2, n3] = np.shape(LEVELSET)

    LEVELSET = LEVELSET.reshape(n1*n2*n3)
    FILL = FILL.reshape(n1*n2*n3)

    todelete = np.where(FILL == False)
    toconcatenate = np.delete(LEVELSET, todelete)
    toconcatenate = toconcatenate.reshape((np.shape(toconcatenate)[0],1))

    CONCAT = np.concatenate((indices_geo, toconcatenate), axis = -1)

    return CONCAT

####################################################################################
############################### ELEMENTS POUR LA REGULARISATION ####################
####################################################################################
def Noyau_gaussien_1D(sigma, nb):
    """Retourne le noyau gaussien 1D"""
    x = np.linspace(-3*sigma, 3*sigma, 2*(nb)+1)
    x1= np.exp(-x**2/(2*sigma**2))
    x1 = x1/np.sum(x1)

    return x1

def Noyau_gaussien_2D(sigma, nb):
    """Retourne le noyau gaussien_2D"""
    x1 = Noyau_gaussien_1D(sigma, nb)

    Fx1 = np.fft.fft(x1)
    Fx1 = Fx1[np.newaxis]
    Fx2 = np.transpose(Fx1)

    Fx1x2 = np.dot(Fx2,Fx1)

    x1x2 = np.real(np.fft.ifft2(Fx1x2))

    return x1x2

def Noyau_gaussien_3D(sigma, nb):
    """Retourne le noyau gaussien_3D"""
    x = np.linspace(-3*sigma, 3*sigma, 2*(nb)+1)
    x1 = Noyau_gaussien_1D(sigma, nb)
    x2x3 = Noyau_gaussien_2D(sigma, nb)

    Fx1 = np.fft.fft(x1)
    Fx1 = Fx1[np.newaxis]

    Fx2x3 = np.fft.fft2(x2x3)
    Fx2x3 = np.transpose(Fx2x3[np.newaxis])

    Fx1x2x3 = np.dot(Fx2x3,Fx1)
    x1x2x3 = np.real(np.fft.ifftn(Fx1x2x3))

    return x1x2x3

def Noyau_laplacien_7():
    """Retourne le noyau du Laplacien discret pour 7 points"""
    return np.array([[[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]],
                    [[0, 1, 0],
                    [1, -6, 1],
                    [0, 1, 0]],
                    [[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]])

def Noyau_laplacien_27():
    """Retourne le noyau du Laplacien discret pour 27 points"""
    return np.array([[[2/26, 3/26, 2/26],
                         [3/26, 6/26, 3/26],
                         [2/26, 3/26, 2/26]],
                        [[3/26, 6/26, 3/26],
                         [6/26, -88/26, 6/26],
                         [3/26, 6/26, 2/26]],
                        [[2/26, 3/26, 2/26],
                         [3/26, 6/26, 2/26],
                         [2/26, 3/26, 2/26]]])

def Convolution(MAT, kernel):
    """Fait la convolution entre la matrice MAT et le noyau kernel"""
    return scipy.signal.oaconvolve(MAT, kernel, mode = "same")

####################################################################################
############################### ELEMENTS POUR LA REGULARISATION ####################
####################################################################################

def Extraction_pcl(CONCAT_LAP, seuil):
    """ Retourne le squelette à partir du laplacien de la fonction distance. On se
    base sur l'extraction des points inférieur à p dans [0,1] fois le min """
    min = np.min(CONCAT_LAP[:,3])
    PCL = CONCAT_LAP[CONCAT_LAP[:,3] <= seuil*min]

    return PCL[:,0:3]
