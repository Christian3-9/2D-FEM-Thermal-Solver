"""
@author: Christian Valencia Narva

"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt


import MultidimensionalSpatialParametricGradient_Solutions as basis
import Gaussian_Quadrature_2D as gq

from enum import Enum

# This class converts faces of an element into a special tagged number
# that is easy to reference and easy to compare against. 
# This should be used in completion of the GetFaceQuadraturePoints
# and in __BdryFaceToVaryingCoordinate__ functions
class BoundaryFace(Enum):
    BOTTOM = 0
    TOP = 1
    LEFT = 2
    RIGHT = 3
    
# Convert a one-dimensional quadrature rule into a quadrature
# rule on a face by extracting out the two-dimensional coordinates
# on the parent domain that correspond to a given face
def GetFaceQuadraturePoints(quad_1d,bdry_face):
    # This section converts 1D quadrature points
    # e.g. on the left face to 1D quadrature points in 2D with parent
    # coordinates of xi = -1 and eta varying using the 1D quadrature
    # similarly for top, bottom, and side coordinates
    # if neither a bottom, top, left, or right is input, this should
    # output an error
    
    if bdry_face == BoundaryFace.BOTTOM:
        quad_pts=quad_1d.quad_pts
        xi_or_etah=-1
        pts=[[pt,xi_or_etah] for pt in quad_pts]
    elif bdry_face == BoundaryFace.TOP:
        quad_pts=quad_1d.quad_pts
        xi_or_etah=1
        pts=[[pt,xi_or_etah] for pt in quad_pts]
    elif bdry_face == BoundaryFace.LEFT:
        quad_pts=quad_1d.quad_pts
        xi_or_etah=-1
        pts=[[xi_or_etah,pt] for pt in quad_pts]
    elif bdry_face == BoundaryFace.RIGHT:
        quad_pts=quad_1d.quad_pts
        xi_or_etah=1
        pts=[[xi_or_etah,pt] for pt in quad_pts]
    else:
        print("Wrong bdry_face")
        return None
    
    
    pts=np.array(pts)
    return pts

        
# Determine which column of a boundary face should be extracted from
# the deformation gradient (Jacobian matrix)
def __BdryFaceToVaryingCoordinate__(bdry_face):
    # This function is to output the index of the appropriate column
    # of the deformation gradient that varies given an input 
    # boundary face. If an invalid face is specified, output an error
    if bdry_face==BoundaryFace.TOP or bdry_face==BoundaryFace.BOTTOM:
        column=0
    elif bdry_face==BoundaryFace.LEFT or bdry_face==BoundaryFace.RIGHT:
        column=1
    return column

    
# Extract the appropriate differential vector from a boundary face
# given points on the face, control points that define the mapping,
# a basis function object, and the face of interest
def DifferentialVector(xi_vals,x_pts,lagrange_basis,bdry_face):
    DF=lagrange_basis.EvaluateDeformationGradient(x_pts,xi_vals)
    DF=np.array(DF)
    column=__BdryFaceToVaryingCoordinate__(bdry_face)
    vector=DF[:,column]
    return vector
    
# Compute the Jacobian of a curve given points on the face,
# control points that define the parent to spatial mapping,
# a basis function object, and the face of interest
def JacobianOneD(xi_vals,x_pts,lagrange_basis,bdry_face):
    J_vector=DifferentialVector(xi_vals,x_pts,lagrange_basis,bdry_face)
    Jacobian=np.sqrt(J_vector[0]**2+J_vector[1]**2)
    return Jacobian
    