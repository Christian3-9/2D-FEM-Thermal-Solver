"""
@author: Christian Valencia Narva

"""

import numpy as np
import MultidimensionalSpatialParametricGradient_Solutions as basis
import Gaussian_Quadrature_2D as gq

def LocalStiffnessMatrix(kappa,basis,x_pts,quadrature):
    
    num_basis_functions=basis.NBasisFuncs()
    KAB=np.zeros((num_basis_functions,num_basis_functions)) #Num of basis functions in each axis

    for i in range(len(quadrature.quad_wts)):
        Jacobian= basis.EvaluateJacobian(x_pts, [*quadrature.quad_pts[i]])
        x_of_e_given_xi=basis.EvaluateSpatialMapping(x_pts, [*quadrature.quad_pts[i]])
        w= quadrature.quad_wts[i]

        for A in range(num_basis_functions):
            Nabla_Na=basis.EvaluateBasisSpatialGradient(A, x_pts, [*quadrature.quad_pts[i]]) 
            Nabla_Na_T=np.transpose(Nabla_Na)

            for B in range(num_basis_functions):
                   
                Nabla_Nb=basis.EvaluateBasisSpatialGradient(B, x_pts, [*quadrature.quad_pts[i]])
                ##I will assume kappa is already a np.array
                KAB[A,B] += w*np.dot(np.dot(Nabla_Na_T,kappa(x_of_e_given_xi[0],x_of_e_given_xi[1])),Nabla_Nb)*Jacobian
    
    return KAB


def LocalForceVector(f,basis,x_pts,quadrature):
    ## I am assuming that we have the same number of interp_pts in both axis
    num_basis_functions=basis.NBasisFuncs()
    F=np.zeros(num_basis_functions)
    for i in range(len(quadrature.quad_wts)):
        Jacobian= basis.EvaluateJacobian(x_pts, [*quadrature.quad_pts[i]])
        x_of_e_given_xi=basis.EvaluateSpatialMapping(x_pts, [*quadrature.quad_pts[i]])
        w= quadrature.quad_wts[i]
        for A in range(num_basis_functions):
            Na=basis.EvalBasisFunction(A,[*quadrature.quad_pts[i]])
            
            F[A] += w*Na*f(x_of_e_given_xi[0],x_of_e_given_xi[1])*Jacobian
    return F
