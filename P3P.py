import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """ 
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
        K:  3x3 numpy array for camera intrisic matrix (given in run_P3P.py)
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 3x1 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])
    f = (K[0,0] + K[1,1])/2  #as given in K matrix
    
    Pt_ind = [0,3,2]
    
    P1 = Pw[Pt_ind[0],:]
    P2 = Pw[Pt_ind[1],:]
    P3 = Pw[Pt_ind[2],:]
    
    f0 = np.transpose(K[0:2,2])
    
    q1 = Pc[Pt_ind[0],:]-f0
    q2 = Pc[Pt_ind[1],:]-f0
    q3 = Pc[Pt_ind[2],:]-f0
    
    # define a,b,c,alpha,beta,gamma
    
    
    a = np.linalg.norm((P2-P3))
    b = np.linalg.norm((P1-P3))
    c = np.linalg.norm((P1-P2))
    
    j1 = 1/np.sqrt(q1[0]**2 + q1[1]**2 + f**2) * np.hstack([q1,f]).reshape(3,1)
    j2 = 1/np.sqrt(q2[0]**2 + q2[1]**2 + f**2) * np.hstack([q2,f]).reshape(3,1)
    j3 = 1/np.sqrt(q3[0]**2 + q3[1]**2 + f**2) * np.hstack([q3,f]).reshape(3,1)
    
    
    alpha_c = np.dot(np.transpose(j2), j3)
    beta_c = np.dot(np.transpose(j1), j3)
    gamma_c = np.dot(np.transpose(j1), j2)
    
    
    # define coefficients of the 4th degree polynomial
    
    A_B = a**2 / b**2
    C_B = c**2 / b**2
    
    A4 = (A_B - C_B -1)**2 - 4* C_B* alpha_c**2
    A3 = 4*((A_B - C_B)* (1 - A_B + C_B) * beta_c - (1 - A_B - C_B)* alpha_c * gamma_c + 2*C_B*alpha_c**2*beta_c)
    A2 = 2*((A_B - C_B)**2 -1 +2* (A_B - C_B)**2 *beta_c**2 + 2*(1 - C_B)*alpha_c**2 - 4* (A_B + C_B) * alpha_c*beta_c*gamma_c+ 2*(1-A_B)*gamma_c**2)
    A1 = 4 * (-(A_B - C_B)*(1+A_B - C_B)*beta_c + 2* A_B *gamma_c**2*beta_c-(1-A_B-C_B)*alpha_c*gamma_c)
    A0 = (1+A_B -C_B)**2 - 4*A_B*gamma_c**2


    # calculate real roots u and v
    
    coeff = np.hstack([A4,A3,A2,A1,A0])
    coeff = coeff.flatten()
    
    v = np.roots(coeff)
    
    S_uv = np.zeros([4,2])
    n_s = 0 #number of solutions for real u,v
    
    for i in range(len(v)):
        if np.isreal(v[i]):
            
            #v_r = np.real(v[i])
            S_uv[n_s,0] = v[i]
            u = (-1 + A_B-C_B)*v[i]**2 - 2*(A_B - C_B) *beta_c *v[i]+1+A_B - C_B
            u = u/(2*(gamma_c-v[i]*alpha_c))
            u = u.flatten()
            S_uv[n_s,1] = np.real(u)
            n_s = n_s +1       
 
    # check for valid distances
    
    V_sol = np.zeros([n_s, 3])
    n_v = 0 # number of solutions for s1,s2,s3>0
    
    for i in range(n_s):
        
        u_temp = S_uv[i,1]
        v_temp = S_uv[i,0]
        
        s1 = c**2 /(1+u_temp**2-2*u_temp*gamma_c)
        s1 = np.sqrt(s1)
        s2 = u_temp * s1
        s3 = v_temp * s1
        
        if s1> 0 and s2>0 and s3>0:
            V_sol[n_v,:] = [s1,s2,s3]
            n_v = n_v + 1
    
    # calculate 3D coordinates in Camera frame
    
    _s1 = V_sol[0,0]
    _s2 = V_sol[0,1]
    _s3 = V_sol[0,2]
    
    _p1 = _s1 * j1
    _p2 = _s2 * j2
    _p3 = _s3 * j3
    
    # Calculate R,t using Procrustes
    
    
    X_pc = np.hstack([_p1,_p2,_p3])
    X_pc = np.transpose(X_pc)
    Y_pw = np.vstack([P1,P2,P3])
    
    R,t = Procruste(X_pc,Y_pw)
    t = t.reshape(3,1)
    
    
    ##### STUDENT CODE END #####
    
    return R,t



def Procruste(X, Y):
    """ 
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate 
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: 1x3 numpy array describing camera translation in the world (t_wc)
        
    """
    
    ##### STUDENT CODE START #####

    R = np.eye(3)
    t = np.zeros([3])
    
    Y = np.transpose(Y)
    X = np.transpose(X)
    
    Y_m = np.mean(Y, axis = 1)
    Y_m = Y_m.reshape(3,1)
    X_m = np.mean(X, axis = 1)
    X_m = X_m.reshape(3,1)
    
    _Y = Y - Y_m
    _X = X - X_m
    
    array = _X @ np.transpose(_Y)
    [U, S , V] = np.linalg.svd(array, full_matrices=True)
    
    V = np.transpose(V)
    D_m = np.eye(3)
    D_m[2,2] = np.linalg.det(V@np.transpose(U))
    R = V@D_m@np.transpose(U)
    
    t = Y_m - R@X_m
    t = t.reshape(1,3)
    


    ##### STUDENT CODE END #####
    
    return R, t
