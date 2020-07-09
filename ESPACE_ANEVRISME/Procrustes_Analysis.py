import numpy as np
from scipy.spatial import procrustes

def Procrustes(Mat_ref, Liste_Mat):
    
    Result_Ref = []
    Result_List = []
    Disparity = []
    for i in Liste_Mat:
        mtx1, mtx2, disparity = procrustes(Mat_ref, i)
        Result_Ref.append(mtx1)
        Result_List.append(mtx2)
        Disparity.append(disparity)
        
    return Result_Ref, Result_List, Disparity
