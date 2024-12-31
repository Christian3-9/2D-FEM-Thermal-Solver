"""
@author: Christian Valencia Narva

"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")

    a_basis=1
    zeta_a=pts[a]
    for zeta_b in pts:
        if zeta_b==zeta_a:
            pass # a term is omitted
        else:
            a_basis=a_basis*(xi-zeta_b)/(zeta_a-zeta_b) #Multiplication to get the a basis
       
    
    return a_basis ##a_th basis function at the location xi


# higher-dimensional basis function with multi-index
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    basis_start=1
    
    for i in range(len(idxs)):
        basis_start=basis_start*LagrangeBasisEvaluation(degs[i],interp_pts[i],xis[i],idxs[i])
        
    return basis_start
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):

    idxs=np.zeros(len(degs))
    for i in range(len(degs)):
        
        if i==0:            
            idxs[i]=A%(degs[i]+1)
            deg_base=(degs[i]+1)
        else:
            idxs[i]=A//deg_base
            deg_base=deg_base*(degs[i]+1)

    idxs=idxs.astype(int)
    
    basis_start=MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis)

    
    return basis_start

def LagrangeBasisParamDervEvaluation(p,pts,xi,a):

    sum=0
    xi_a=pts[a]
    for j in range(len(pts)):
        basis=1
        xi_j=pts[j]

        if xi_a==xi_j:
            pass
        else:
            for b in range(len(pts)):
                xi_b=pts[b]
                if xi_a==xi_b or xi_j==xi_b:
                    pass
                else:
                    basis=basis*(xi-xi_b)/(xi_a-xi_b)
    
            sum=sum+1/(xi_a-xi_j)*basis

    derivative=sum
    return derivative

def GlobalToLocalIdxs(A,degs): ##IS THIS IEN?
    idxs=np.zeros(len(degs))
    for i in range(len(degs)):
        
        if i==0:            
            idxs[i]=A%(degs[i]+1)
            deg_base=(degs[i]+1)
        else:
            idxs[i]=A//deg_base
            deg_base=deg_base*(degs[i]+1)

    idxs=idxs.astype(int)
    return idxs


# This is a class that describes a Lagrange basis
# in two dimensions
class LagrangeBasis2D:
    
    # initializor
    def __init__(self,degx,degy,interp_pts_x,interp_pts_y):
        self.degs = [degx,degy]
        self.interp_pts = [interp_pts_x,interp_pts_y]
        self.side_bfs={}
        
    # the number of basis functions is the 
    # product of basis functions in the x (xi)
    # and y (eta) directions
    def NBasisFuncs(self):
        xi_pol_degs=self.degs[0]
        eta_pol_degs=self.degs[1]
        total_NBasisFuncs=(xi_pol_degs+1)*(eta_pol_degs+1)
        return total_NBasisFuncs

    def EvalBasisFunction(self,A,xi_vals):
        return MultiDimensionalBasisFunction(A,self.degs,self.interp_pts,xi_vals)    
    

    def EvalBasisDerivative(self,A,xis,dim):

        idxs=GlobalToLocalIdxs(A,self.degs)

        partial_derviative=LagrangeBasisParamDervEvaluation(self.degs[dim],self.interp_pts[dim],xis[dim],idxs[dim])
        product=1
        for i in range(len(idxs)):
            
            if i==dim:
                pass
            else:
                product=product*LagrangeBasisEvaluation(self.degs[i],self.interp_pts[i],xis[i],idxs[i])

        partial_derviative=partial_derviative*product
        
        return partial_derviative


    # Evaluate a sum of basis functions times 
    # coefficients on the parent domain
    def EvaluateFunctionParentDomain(self, d_coeffs, xi_vals):
        nbfs=self.NBasisFuncs()
        u=0
        for a in range(nbfs):
           u=u+d_coeffs[a]*self.EvalBasisFunction(a,xi_vals) 
        return u
        
    # Evaluate the spatial mapping from xi and eta
    # into x and y coordinates
    def EvaluateSpatialMapping(self, x_pts, xi_vals):
        nbfs=len(x_pts)
        x_e=0
        y_e=0
        for a in range(nbfs):
            x_e=x_e+x_pts[a][0]*self.EvalBasisFunction(a,xi_vals)
            y_e=y_e+x_pts[a][1]*self.EvalBasisFunction(a,xi_vals)
        
        return [x_e,y_e]
    
    # Evaluate the Deformation Gradient (i.e.
    # the Jacobian matrix)
    def EvaluateDeformationGradient(self, x_pts, xi_vals):
        sum11=0
        sum12=0
        sum21=0
        sum22=0
        for a in range(len(x_pts)):   
            sum11=sum11+x_pts[a][0]*self.EvalBasisDerivative(a,xi_vals,0)
            sum12=sum12+x_pts[a][0]*self.EvalBasisDerivative(a,xi_vals,1)
            sum21=sum21+x_pts[a][1]*self.EvalBasisDerivative(a,xi_vals,0)
            sum22=sum22+x_pts[a][1]*self.EvalBasisDerivative(a,xi_vals,1)

        partial_x_partial_xi=sum11
        partial_x_partial_eta=sum12
        partial_y_partial_xi=sum21
        partial_y_partial_eta=sum22
        DF=[[partial_x_partial_xi,partial_x_partial_eta],[partial_y_partial_xi,partial_y_partial_eta]]
        #print(DF)
        return np.array(DF)
    
    # Evaluate the jacobian (or the determinant
    # of the deformation gradient)
    def EvaluateJacobian(self, x_pts, xi_vals):
        DF=self.EvaluateDeformationGradient(x_pts,xi_vals)
        Determinant=DF[0][0]*DF[1][1]-DF[1][0]*DF[0][1]
        return Determinant

    # Evaluate the parametric gradient of a basis
    # function
    def EvaluateBasisParametricGradient(self,A, xi_vals):
        partial_basis_partial_xi=self.EvalBasisDerivative(A,xi_vals,0)
        partial_basis_partial_eta=self.EvalBasisDerivative(A,xi_vals,1)
        pNa_pxi=partial_basis_partial_xi
        PNa_peta=partial_basis_partial_eta
        return np.array([pNa_pxi,PNa_peta])

    # Evaluate the parametric gradient of a basis
    # function
    def EvaluateBasisSpatialGradient(self,A, x_pts, xi_vals):
        DF=np.array(self.EvaluateDeformationGradient(x_pts,xi_vals))
        DF_minus=np.linalg.inv(DF) 
        DF_minus_transpose=DF_minus.transpose()
        parametricGradient=np.array(self.EvaluateBasisParametricGradient(A,xi_vals))
        SpatialGradient=DF_minus_transpose@parametricGradient
        return SpatialGradient
    

    # Grid plotting functionality that is used
    # in all other plotting functions
    def PlotGridData(self,X,Y,Z,npts=21,contours=False,xlabel=r"$x$",ylabel=r"$y$",zlabel=r"$z$", show_plot = True):
        if contours:
            fig, ax = plt.subplots()
            surf = ax.contourf(X,Y,Z,levels=100,cmap=matplotlib.cm.jet)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(surf)
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.jet,
                           linewidth=0, antialiased=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        if show_plot:
            plt.show()
        
        return fig,ax

    # plot the mapping from parent domain to 
    # spatial domain            
    def PlotSpatialMapping(self,x_pts,npts=21,contours=False):
        dim = len(x_pts[0])
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)

        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                if dim == 3:
                    Z[i,j] = pt[2] 
        
        self.PlotGridData(X,Y,Z,contours=contours,)

    # plot a basis function defined on a parent
    # domain              
    def PlotBasisFunctionParentDomain(self,A,npts=21,contours=False):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)

        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")

    # plot a basis function defined on a spatial
    # domain
    def PlotBasisFunctionSpatialDomain(self,A,x_pts,npts=21,contours=False,on_parent_domain=True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)

        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)
        
        self.PlotGridData(X,Y,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")


    # plot a solution field defined on a parent
    # domain
    def PlotParentSolutionField(self,d_coeffs,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                Z[j,i] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$u_h^e(\xi,\eta)$")

    # define a solution field mapped into the
    # spatial domain for an element
    def PlotSpatialSolutionField(self,d_coeffs,x_pts,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                Z[j,i] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$u_h^e(x,y)$")


    # plot Jacobians defined on the spatial 
    # or parent domain
    def PlotJacobian(self,x_pts,npts=21,contours = False, parent_domain = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                if not parent_domain:
                    pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                    X[j,i] = pt[0]
                    Y[j,i] = pt[1]
                Z[j,i] = self.EvaluateJacobian(x_pts,xi_vals)
    
        if parent_domain:
            self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$")
        else:
            self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$J^e(x,y)$")

    def PlotBasisFunctionGradient(self,A,x_pts,npts=21, parent_domain = True, parent_gradient = True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        U = np.zeros(Xi.shape)
        V = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                if not parent_domain:
                    pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                    X[j,i] = pt[0]
                    Y[j,i] = pt[1]
                if parent_gradient:
                    grad = self.EvaluateBasisParametricGradient(A, xi_vals)
                else:
                    grad = self.EvaluateBasisSpatialGradient(A, x_pts, xi_vals)
                U[j,i] = grad[0]
                V[j,i] = grad[1]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)

        if parent_domain:
            fig,ax = self.PlotGridData(Xi,Eta,Z,contours=True,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$",show_plot = False)
            ax.quiver(Xi,Eta,U,V)
        else:
            fig,ax = self.PlotGridData(X,Y,Z,contours=True,zlabel=r"$J^e(x,y)$",show_plot = False)
            ax.quiver(X,Y,U,V)
        plt.show()
