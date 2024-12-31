"""
@author: Christian Valencia Narva

"""

import numpy as np
import sys


import MultidimensionalSpatialParametricGradient_Solutions as basis
import Gaussian_Quadrature_2D as quad
import Quadrature_Operations as q1d 
import Boundary_Conditions as bnd 


def LocalStiffnessBoundaryTerms(basis,bdry_data,x_pts, quadrature_1d):
    num_basis_functions=basis.NBasisFuncs()
    KAB=np.zeros((num_basis_functions,num_basis_functions)) #Num of basis functions in each axis

    if bdry_data[0][1].type != bnd.BoundaryConditionType.ROBIN:

        return KAB    #We return a matrix full of zeros which won't change our calculations.

    else:

        face_parametric_pts=q1d.GetFaceQuadraturePoints(quadrature_1d,bdry_data[0][0])
        for i in range(len(quadrature_1d.quad_wts)):
            Jacobian=q1d.JacobianOneD(face_parametric_pts[i],x_pts,basis,bdry_data[0][0])
            x_of_e_given_xi=basis.EvaluateSpatialMapping(x_pts, face_parametric_pts[i])
            u_multiplier=bdry_data[0][1].u_multiplier(x_of_e_given_xi[0],x_of_e_given_xi[1])
            weight=quadrature_1d.quad_wts[i]
            for A in range(num_basis_functions):
                NA=basis.EvalBasisFunction(A,face_parametric_pts[i])
                for B in range(num_basis_functions):
                    NB=basis.EvalBasisFunction(B,face_parametric_pts[i]) 
                    KAB[A,B] += weight*NA*u_multiplier*NB*Jacobian

    return KAB


def LocalForceBoundaryTerms(basis,bdry_data,x_pts,ke_init, quadrature_1d, local_global):
    num_basis_functions=basis.NBasisFuncs()
    F=np.zeros(num_basis_functions)

    #Case of robin
    if bdry_data[0][1].type == bnd.BoundaryConditionType.ROBIN:
        F=np.zeros(num_basis_functions)
        face_parametric_pts=q1d.GetFaceQuadraturePoints(quadrature_1d,bdry_data[0][0])
        for i in range(len(quadrature_1d.quad_wts)):
            Jacobian=q1d.JacobianOneD(face_parametric_pts[i],x_pts,basis,bdry_data[0][0])
            x_of_e_given_xi=basis.EvaluateSpatialMapping(x_pts, face_parametric_pts[i])
            rhs_func=bdry_data[0][1].rhs_func(x_of_e_given_xi[0],x_of_e_given_xi[1])
            weight=quadrature_1d.quad_wts[i]
            for A in range(num_basis_functions):
                NA=basis.EvalBasisFunction(A,face_parametric_pts[i]) 
                F[A] += weight*NA*rhs_func*Jacobian
        

        return F


    elif bdry_data[0][1].type == bnd.BoundaryConditionType.NEUMANN:
        F=np.zeros(num_basis_functions)
        face_parametric_pts=q1d.GetFaceQuadraturePoints(quadrature_1d,bdry_data[0][0])
        for i in range(len(quadrature_1d.quad_wts)):
            Jacobian=q1d.JacobianOneD(face_parametric_pts[i],x_pts,basis,bdry_data[0][0])
            x_of_e_given_xi=basis.EvaluateSpatialMapping(x_pts, face_parametric_pts[i])
            flux=bdry_data[0][1].rhs_func(x_of_e_given_xi[0],x_of_e_given_xi[1])
            weight=quadrature_1d.quad_wts[i]
            for A in range(num_basis_functions):
                NA=basis.EvalBasisFunction(A,face_parametric_pts[i]) 
                F[A] += weight*NA*flux*Jacobian

        return F

    elif bdry_data[0][1].type == bnd.BoundaryConditionType.DIRICHLET:

        idxs_side=basis.side_bfs[bdry_data[0][0]]
        face_parametric_pts=idxs_side
        for i in range(len(idxs_side)):     
            x_of_e_given_xi=x_pts[face_parametric_pts[i]]

            g_b_hat=bdry_data[0][1].rhs_func(x_of_e_given_xi[0],x_of_e_given_xi[1])
            index_to_store_dict=local_global[face_parametric_pts[i]]
            bdry_data[0][1].bdry_coeffs[index_to_store_dict]=g_b_hat
            F -= g_b_hat*ke_init[:,face_parametric_pts[i]]
            
        return F
        

