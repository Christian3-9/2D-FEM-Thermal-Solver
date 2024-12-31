"""
@author: Christian Valencia Narva

"""

import math
import numpy as np
import scipy
from scipy import sparse
from scipy import linalg

import sys
import itertools

# Beta term from Trefethen, Bau Equation 37.6
def BetaTerm(n):
    if n <= 0:
        return 0
    else:
        return 0.5*math.pow((1-math.pow(2*n,-2)),-0.5)

# Theorem 37.4 from Trefethen, Bau
def ComputeQuadraturePtsWts(n):
    # Compute the Jacobi Matrix, T_n
    # given explicitly in Equation 37.6
    diag = np.zeros(n)
    off_diag = np.zeros(n-1)
    for i in range(0,n-1):
        off_diag[i] = BetaTerm(i+1)
        
    # Divide and conquer algorithm for tridiagonal
    # matrices
    # w is eigenvalues
    # v is matrix with columns corresponding eigenvectors
    [w,v] = scipy.linalg.eigh_tridiagonal(diag,off_diag,check_finite=False)
    
    # nodes of quadrature given as eigenvalues
    nodes = w
    # weights given as two times the square of the first 
    # index of each eigenvector
    weights = 2*(v[0,:]**2)
    
    return [nodes,weights]

class GaussQuadrature1D:
    
    def __init__(self,n_quad, start_pt = -1, end_pt = 1):
        self.n_quad = n_quad
        [self.quad_pts,self.quad_wts] = ComputeQuadraturePtsWts(self.n_quad)
        self.jacobian = 1
        
        if start_pt != -1 or end_pt != 1:
           self.__TransformToInterval__(start_pt,end_pt)
           
     
    def __TransformToInterval__(self,start,end):
        # complete this function
        self.quad_pts= (end-start)/2*self.quad_pts+(end+start)/2
        self.jacobian=(end-start)/2

        

class GaussQuadratureQuadrilateral:
    
    def __init__(self,n_quad,start = -1,end = 1):
        self.n_quad = n_quad
        self.jacobian = 1
        [self.quad_pts,self.quad_wts] = ComputeQuadraturePtsWts(self.n_quad)
        self.start = start
        self.end = end
        if start != -1 or end != 1:
            self.__TransformToInterval__(start,end)
            self.jacobian=(end-start)/2*(end-start)/2 #Two dimensions and since they have the same number of deg is just J^2

        quad_wts_list=[]
        quad_pts_list=[]

        for i in range(n_quad):
            for j in range(n_quad):
                quad_wts_list.append(float(self.quad_wts[i])*float(self.quad_wts[j]))
                quad_pts_list.append((float(self.quad_pts[i]),float(self.quad_pts[j])))

        self.quad_wts=quad_wts_list
        self.quad_pts=quad_pts_list

    def __TransformToInterval__(self,start,end):
        self.quad_pts= (end-start)/2*self.quad_pts+(end+start)/2
        