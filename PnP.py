import numpy as np
from est_homography import est_homography

def PnP(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
        K:  3x3 numpy array for camera intrisic matrix (given in run_PnP.py)
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####


    R = np.eye(3)
    t = np.zeros([3])
    
    #calculating H
    
    Pw = Pw[:,0:2]
    H = est_homography(Pw,Pc)
    inv_K = np.linalg.inv(K)
    
    # calculating R & T
    _h = np.matmul(inv_K, H)
    
    _h1 = _h[:,0]
    _h2 = _h[:,1]
    _h3 = _h[:,2]
    
    array = _h.copy()
    array[:,2] = np.cross(_h1,_h2)
     
    
    [U, S , V] = np.linalg.svd(array, full_matrices=True)
    
    
    temp = np.eye(3)
    
    #V = np.transpose(V)
    temp[2,2] = np.linalg.det(U@V)
    
    R = U@temp@V
    
    l_h1 = np.linalg.norm(_h1)
    T = _h3/l_h1
    
    #transpose
    
    R = np.transpose(R)
    
    t = -R@T
    t = t.reshape(3,1)
    
    
    ##### STUDENT CODE END #####

    return R, t
