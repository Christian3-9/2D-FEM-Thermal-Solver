"""
@author: Christian Valencia Narva

"""

import numpy as np
import scipy
import sys
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import Gaussian_Quadrature_2D as quad
import MultidimensionalSpatialParametricGradient_Solutions as bf
import Surfacemesh as mesh
import Basismesh as basismesh
import Boundary_Conditions as bnd
import Multidimensional_Local_Assembly_Routines as la
import Quadrature_Operations as q1d
import LocalStiffnessBoundaries as labnd


def Brick_Simulation(mode):
    # load input geometry
    mesh_obj = r"brick_obj.obj" #input mesh file
    bdry_obj = r"brick_boundaries.obj" #input boundary data
    surf_mesh = mesh.SurfaceMesh.FromOBJ_FileName(mesh_obj)
    # Alternatively, enter brick_boundaries.obj into AppendSurfaceBoundarySets
    mesh_bndry_data = surf_mesh.AppendSurfaceBoundarySets(bdry_obj)

    # define input basis functions
    degx = mode
    degy = mode
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
    basis = bf.LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
 
    
    # combine the basis functions with the input geometry
    bm = basismesh.BasisMesh(surf_mesh,basis)
    # extract out new boundary data
    bndry_data = bm.boundaries
    
    # Define quadrature routines
    n_quad = 2 * max(degx,degy)
    gauss = quad.GaussQuadratureQuadrilateral(n_quad,-1,1)
    gauss_1d = quad.GaussQuadrature1D(n_quad,-1,1)
    
    # define input force functions, boundary conditions,
    # and other parameters
    f = lambda x,y: 0
    g = lambda x,y: 10
    h = lambda x,y: 0.1
    u_robin_mult = lambda x,y: -9.06
    robin_rhs = lambda x,y: -9.06*20
    kappa = Kappa                                     
    
    # assign boundary conditions to various boundaries;
    # this particular code assigns each boundary condition to a
    # different boundary component (arbitrarily)
    boundaries = []
    count = 0

    # print(len(bndry_data)) ## We have three but we have 4 different types of
    for boundary in bndry_data:
        if count == 0:
            new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN, boundary)

        elif count == 1:
            new_bnd = bnd.RobinBoundaryCondition(bnd.BoundaryConditionType.ROBIN,boundary,u_robin_mult,robin_rhs)

        elif count == 2:
            new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN, boundary)
        
        elif count == 3:
            new_bnd = bnd.DirichletBoundaryCondition(bnd.BoundaryConditionType.DIRICHLET, boundary, g)
        
        elif count == 4:
            new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN, boundary,h)

        elif count == 5:
            new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN, boundary,h)

        count += 1
        boundaries.append(new_bnd)
    
 
    # solve the heat conduction problem
    d = HeatConductionProblem(bm, gauss, gauss_1d, f, kappa, boundaries)
    
    # only sample coefficients on the original mesh
    # because we lack good ways to plot high-order polynomials
    d_for_plotting = ExtractDForPlotting(bm, d)
    print(max(d))


    ############ FLUX CALCULATIONS ON BOUNDARIES ##########
    pts=bm.pts
    n_right=np.array([1,0])
    n_top=np.array([0,1])
    n_left=np.array([-1,0])
    n_bot=np.array([0,-1])

    ##################################
    ## If BIQUADRATIC METHOD IS APPLIED
    if degx == 2 and degy == 2 :
        flux_right=0
        flux_top=0
        flux_left=0
        flux_bot=0

        count_right=0
        count_top=0
        count_left=0
        count_bot=0
        for e in range(len(bm.IEN)):
            nodes = bm.IEN[e]
            if sum(node in bm.boundaries_lists[0] for node in nodes) >= 2: #RIGHT and ensuring at least two nodes in the bdry
                x_pts = [pts[i] for i in bm.IEN[e]]
                for nq in range(len(gauss.quad_wts)):
                    count_right+=1
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, *[gauss.quad_pts[nq]])
                    grad_u=EvaluateSolutionGradient(e, bm.basis, d, bm.IEN, DF, *[gauss.quad_pts[nq]])
                    flux= kappa(0,0)@grad_u
                    flux_right += np.dot(flux,n_right) 

            elif sum(node in bm.boundaries_lists[1] for node in nodes) >= 2: #TOP and ensuring at least two nodes in the bdry
                x_pts = [pts[i] for i in bm.IEN[e]]
                for nq in range(len(gauss.quad_wts)):
                    count_top+=1
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, *[gauss.quad_pts[nq]])
                    grad_u=EvaluateSolutionGradient(e, bm.basis, d, bm.IEN, DF, *[gauss.quad_pts[nq]])
                    flux= kappa(0,0)@grad_u
                    flux_top += np.dot(flux,n_top)
            
            elif sum(node in bm.boundaries_lists[2] for node in nodes) >= 2: #LEFT and ensuring at least two nodes in the bdry
                x_pts = [pts[i] for i in bm.IEN[e]]
                for nq in range(len(gauss.quad_wts)):
                    count_left+=1
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, *[gauss.quad_pts[nq]])
                    grad_u=EvaluateSolutionGradient(e, bm.basis, d, bm.IEN, DF, *[gauss.quad_pts[nq]])
                    flux= kappa(0,0)@grad_u
                    flux_left += np.dot(flux,n_left)

            elif sum(node in bm.boundaries_lists[3] for node in nodes) >= 2: #TOP and ensuring at least two nodes in the bdry
                x_pts = [pts[i] for i in bm.IEN[e]]
                for nq in range(len(gauss.quad_wts)):
                    count_bot+=1
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, *[gauss.quad_pts[nq]])
                    grad_u=EvaluateSolutionGradient(e, bm.basis, d, bm.IEN, DF, *[gauss.quad_pts[nq]])
                    flux= kappa(0,0)@grad_u
                    flux_bot += np.dot(flux,n_bot)


        print("Deg x and Deg y =1 -> flux right", flux_right/count_right)
        print("Deg x and Deg y =1 -> flux top", flux_top/count_top)
        print("Deg x and Deg y =1 -> flux left", flux_left/count_left)
        print("Deg x and Deg y =1 -> flux bot", flux_bot/count_bot)  

    #########################
    ## If BILINEAR METHOD IS APPLIED
    if degx==1 and degy==1:
        bdry_vs = __ExtractSideVertices__(bm.basis)
        
        flux_sum_right=0
        length_right=0
        for e in boundaries[0].element_IEN:
            elem_bdries = __ExtractElementBoundaries__(e,bm.IEN,bdry_vs,boundaries)
            x_pts = [pts[i] for i in bm.IEN[e]]
            
            length_local=0
            for nq in range(len(gauss_1d.quad_wts)):
                    xi_vals=q1d.GetFaceQuadraturePoints(gauss_1d,elem_bdries[0][0])
                    Jacobian=q1d.JacobianOneD(xi_vals[nq],x_pts,bm.basis,elem_bdries[0][0])               
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, xi_vals[nq])
                    grad_u=EvaluateSolutionGradient(e, bm.basis, d, bm.IEN, DF, xi_vals[nq])
                    flux= kappa(0,0)@grad_u
                    flux_sum_right += np.dot(flux,n_right)*Jacobian*gauss_1d.quad_wts[nq]
                    #pts_length=q1d.GetFaceQuadraturePoints(gauss_1d,elem_bdries[0][0])
                    
                    length_local += gauss.quad_wts[nq]*Jacobian
            length_right+= length_local


        flux_sum_top=0
        length_top=0
        for e in boundaries[1].element_IEN:
            elem_bdries = __ExtractElementBoundaries__(e,bm.IEN,bdry_vs,boundaries)
            x_pts = [pts[i] for i in bm.IEN[e]]
            
            length_local=0
            for nq in range(len(gauss_1d.quad_wts)):
                    xi_vals=q1d.GetFaceQuadraturePoints(gauss_1d,elem_bdries[0][0])
                    Jacobian=q1d.JacobianOneD(xi_vals[nq],x_pts,bm.basis,elem_bdries[0][0])               
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, xi_vals[nq])
                    grad_u=EvaluateSolutionGradient(e, bm.basis, d, bm.IEN, DF, xi_vals[nq])
                    flux= kappa(0,0)@grad_u
                    flux_sum_top += np.dot(flux,n_top)*Jacobian*gauss_1d.quad_wts[nq]
                    #pts_length=q1d.GetFaceQuadraturePoints(gauss_1d,elem_bdries[0][0])
                    
                    length_local += gauss.quad_wts[nq]*Jacobian
            length_top+= length_local

        flux_sum_left=0
        length_left=0
        for e in boundaries[2].element_IEN:
            elem_bdries = __ExtractElementBoundaries__(e,bm.IEN,bdry_vs,boundaries)
            x_pts = [pts[i] for i in bm.IEN[e]]
            
            length_local=0
            for nq in range(len(gauss_1d.quad_wts)):
                    xi_vals=q1d.GetFaceQuadraturePoints(gauss_1d,elem_bdries[0][0])
                    Jacobian=q1d.JacobianOneD(xi_vals[nq],x_pts,bm.basis,elem_bdries[0][0])               
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, xi_vals[nq])
                    grad_u=EvaluateSolutionGradient(e, bm.basis, d, bm.IEN, DF, xi_vals[nq])
                    flux= kappa(0,0)@grad_u
                    flux_sum_left += np.dot(flux,n_left)*Jacobian*gauss_1d.quad_wts[nq]
                    #pts_length=q1d.GetFaceQuadraturePoints(gauss_1d,elem_bdries[0][0])
                    
                    length_local += gauss.quad_wts[nq]*Jacobian
            length_left+= length_local

        flux_sum_bot=0
        length_bot=0
        for e in boundaries[3].element_IEN:
            elem_bdries = __ExtractElementBoundaries__(e,bm.IEN,bdry_vs,boundaries)
            x_pts = [pts[i] for i in bm.IEN[e]]
            
            length_local=0
            for nq in range(len(gauss_1d.quad_wts)):
                    xi_vals=q1d.GetFaceQuadraturePoints(gauss_1d,elem_bdries[0][0])
                    Jacobian=q1d.JacobianOneD(xi_vals[nq],x_pts,bm.basis,elem_bdries[0][0])               
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, xi_vals[nq])
                    grad_u=EvaluateSolutionGradient(e, bm.basis, d, bm.IEN, DF, xi_vals[nq])
                    flux= kappa(0,0)@grad_u
                    flux_sum_bot += np.dot(flux,n_bot)*Jacobian*gauss_1d.quad_wts[nq]
                    #pts_length=q1d.GetFaceQuadraturePoints(gauss_1d,elem_bdries[0][0])
                    
                    length_local += gauss.quad_wts[nq]*Jacobian
            length_bot+= length_local

        q_right_avg=flux_sum_right/length_right 
        print("Deg x and Deg y =1 -> flux right = ",q_right_avg)
        print("LENGTH",length_right )

        q_top_avg=flux_sum_top/length_top 
        print("Deg x and Deg y =1 -> flux top = ",q_top_avg)
        print("LENGTH",length_top )


        q_left_avg=flux_sum_left/length_left 
        print("Deg x and Deg y =1 -> flux left = ",q_left_avg)
        print("LENGTH",length_left )


        q_bot_avg=flux_sum_bot/length_bot 
        print("Deg x and Deg y =1 -> flux bot = ",q_bot_avg)
        print("LENGTH",length_bot )


    #########################

    # plot the solution
    mesh.PlotTriangulationSolution(surf_mesh, d_for_plotting)
    plt.show()

    
# This is an example of how to run the code
# if you make all boundaries Dirichlet with g=x-y, you should get 
# contour plots that look linear in the diagonal direction
# to test that your functions are working correctly
def TestingProblem_block():

    mesh_obj = "unit_square.obj" #input mesh file
    bdry_obj = "unit_square_boundaries.obj" #input boundary data

    surf_mesh = mesh.SurfaceMesh.FromOBJ_FileName(mesh_obj)
    # Alternatively, enter brick_boundaries.obj into AppendSurfaceBoundarySets
    mesh_bndry_data = surf_mesh.AppendSurfaceBoundarySets(bdry_obj)

    # define input basis functions
    degx = 1
    degy = 1
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
    basis = bf.LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # combine the basis functions with the input geometry
    bm = basismesh.BasisMesh(surf_mesh,basis)
    # extract out new boundary data
    bndry_data = bm.boundaries
    print("HERE", bm.boundaries)
    print("list of boundaries",bm.boundaries_lists)
    print(bm.mesh.boundary_set_vert_idxs)
    print(bm.IEN)
    # Define quadrature routines
    n_quad = 2 * max(degx,degy)
    gauss = quad.GaussQuadratureQuadrilateral(n_quad,-1,1)
    gauss_1d = quad.GaussQuadrature1D(n_quad,-1,1)
    
    # define input force functions, boundary conditions,
    # and other parameters
    f = lambda x,y: 0
    g = lambda x,y: 0
    h_right = lambda x,y: -np.pi*np.sin(np.pi*y)
    h_top= lambda x,y: -np.pi*np.sin(np.pi*x)
    f= lambda x,y: 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
    kappa = Kappa                                     
    
    # assign boundary conditions to various boundaries;
    # this particular code assigns each boundary condition to a
    # different boundary component (arbitrarily)
    boundaries = []
    count = 0

    for boundary in bndry_data: ## 0 IS TOP, 1 IS RIGH, 2 IS BOTTOM, 3 IS LEFT
        if count == 0:
            new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN, boundary,h_top)

        elif count == 1:
            new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN, boundary,h_right)


        elif count == 2:
            new_bnd = bnd.DirichletBoundaryCondition(bnd.BoundaryConditionType.DIRICHLET, boundary, g)

        
        elif count == 3:
            new_bnd = bnd.DirichletBoundaryCondition(bnd.BoundaryConditionType.DIRICHLET, boundary, g)

        count += 1
        boundaries.append(new_bnd)



    # solve the heat conduction problem
    d = HeatConductionProblem(bm, gauss, gauss_1d, f, kappa, boundaries)
    
    # only sample coefficients on the original mesh
    # because we lack good ways to plot high-order polynomials
    d_for_plotting = ExtractDForPlotting(bm, d)
    print(max(d))
    # plot the solution
    mesh.PlotTriangulationSolution(surf_mesh, d_for_plotting)
    plt.show()




# Thermal conductivity
def Kappa(x, y):
    return np.eye(2)*3.3

# ID array for (potentially) high-order polynomials
def GetIDArray(bm, boundaries):
    dirichlet_nodes = set()    

    for boundary in boundaries:
        if boundary.type == bnd.BoundaryConditionType.DIRICHLET:
            dirichlet_nodes.update(boundary.bdry_nodes)
    
    n_bf = len(bm.pts)
    
    counter = 0
    ID = np.zeros(n_bf,dtype=int)
    for i in range(0,n_bf):
        if i in dirichlet_nodes:
            ID[i] = -1
        else:
            ID[i] = counter
            counter += 1
    
    return ID

# IEN array from the basis mesh
def GetIENArray(bm):
    return bm.IEN
    
# Extract the basis function indices of an element
# corresponding to various sides
# these are local indices (between 0 and deg^2-1)
def AppendSideBases(basis):
    basis.side_bfs[q1d.BoundaryFace.LEFT]=[(basis.degs[0]+1)*i for i in range(0,basis.degs[1]+1)]
    basis.side_bfs[q1d.BoundaryFace.RIGHT]=[(basis.degs[0]+1)*(i+1)-1 for i in range(0,basis.degs[1]+1)]
    basis.side_bfs[q1d.BoundaryFace.BOTTOM]=[i for i in range(0,basis.degs[1]+1)]
    basis.side_bfs[q1d.BoundaryFace.TOP]=[(basis.degs[0]+1)*basis.degs[1] + i for i in range(0,basis.degs[1]+1)]

# Extract the corner vertices of a mesh element on each side
# This is used in determining if the side of a
# mesh element intersects a boundary in 
# __ExtractElementBoundaries__
def __ExtractSideVertices__(basis):
    numBasis=basis.NBasisFuncs()
    numBasis_edge=int(np.sqrt(numBasis))

    edge1=[0,numBasis_edge-1] ##BOTTOM
    edge2=[0,numBasis-numBasis_edge] ## LEFT
    edge3=[numBasis-numBasis_edge,numBasis-1] ## TOP
    edge4=[numBasis-1,numBasis_edge-1] ##RIGHT

    edgeList=[edge1,edge2,edge3,edge4]

    return edgeList
    
# Determine which boundaries intersect this mesh element
# bdry_vs comes from __ExtractSideVertices__
# output a list that contains lists of 
#       (BoundaryFaces that intersect a given boundary (idx 0) and
#           the boundary that is intersected (idx 1))
# This is input for Local Stiffness and Force evaluations on boundaries
def __ExtractElementBoundaries__(e,IEN,bdry_vs,boundaries):
    elementBoundaries=[]

    for i in range(len(boundaries)):
        if IEN[e][bdry_vs[0][0]] in boundaries[i].bdry_nodes and IEN[e][bdry_vs[0][1]] in boundaries[i].bdry_nodes:
            elementBoundaries.append([q1d.BoundaryFace.BOTTOM,boundaries[i]])
            boundaries[i].element_IEN.add(e)
        elif IEN[e][bdry_vs[1][0]] in boundaries[i].bdry_nodes and IEN[e][bdry_vs[1][1]] in boundaries[i].bdry_nodes:
            elementBoundaries.append([q1d.BoundaryFace.LEFT,boundaries[i]])
            boundaries[i].element_IEN.add(e)
        elif IEN[e][bdry_vs[2][0]] in boundaries[i].bdry_nodes and IEN[e][bdry_vs[2][1]] in boundaries[i].bdry_nodes:
            elementBoundaries.append([q1d.BoundaryFace.TOP,boundaries[i]])
            boundaries[i].element_IEN.add(e)
        elif IEN[e][bdry_vs[3][0]] in boundaries[i].bdry_nodes and IEN[e][bdry_vs[3][1]] in boundaries[i].bdry_nodes:
            elementBoundaries.append([q1d.BoundaryFace.RIGHT,boundaries[i]])
            boundaries[i].element_IEN.add(e)

    return elementBoundaries 
    
# Determine all coeffiecients multiplying basis functions
# (including both known and unknown)
def ExtractTotalD(ID,d,boundaries):
    j=0 
    d_total=np.zeros(len(ID))
    for i in range(len(ID)):
        #j=ID[i]
        if ID[i]==-1:
            for bdry in boundaries:
                if bdry.type == bnd.BoundaryConditionType.DIRICHLET and i in bdry.bdry_coeffs:
                    d_total[i]=bdry.bdry_coeffs[i]

        else:
            d_total[i]=d[ID[i]]
            j+=1

    return d_total
    

# extract out only coefficients corresponding to
# the original mesh vertices
def ExtractDForPlotting(bm,d):
    return d[:len(bm.mesh.vs)]
    
# Solve the 2D Poisson equation
def HeatConductionProblem(bm, gauss, gauss_1d, f, kappa, boundaries):
        
    ID = GetIDArray(bm,boundaries)
    IEN = GetIENArray(bm)
    
    pts = bm.pts
    
    n = max(ID)+1
    
    # These are input for scipy.sparse.coo_matrix data
    k_row_idx = []
    k_col_idx = []
    k_data = []    
    F = np.zeros(n)
    
    n_local = bm.basis.NBasisFuncs()

    
    AppendSideBases(bm.basis)
    # define the local left, right, top, and bottom vertices
    bdry_vs = __ExtractSideVertices__(bm.basis)
    

    for e in range(0,len(bm.IEN)):
        # extract control points associated with this element
        x_pts = [pts[i] for i in bm.IEN[e]]
        ke = la.LocalStiffnessMatrix(kappa, bm.basis, x_pts, gauss)
        fe = la.LocalForceVector(f, bm.basis, x_pts, gauss)
        
        # determine if it is on the boundary; if so, perform appropriate boundary operations
        elem_bdries = __ExtractElementBoundaries__(e,IEN,bdry_vs,boundaries) #This tells us 
        if len(elem_bdries) > 0:
            for i in range(len(elem_bdries)):
                # Update stiffness matrix first to take care of Robin boundary conditions
                ke += labnd.LocalStiffnessBoundaryTerms(bm.basis, [elem_bdries[i]], x_pts, gauss_1d)
                # Because we've taken care of the entire element,
                # we can now add in the local force and anything
                # due to Robin-Dirichlet interactions is properly
                # taken care of
                fe += labnd.LocalForceBoundaryTerms(bm.basis, [elem_bdries[i]], x_pts, ke, gauss_1d, IEN[e])
                
        # Assemble into global matrix
        # Loop over elements
        for i, node_i in enumerate(IEN[e]):  # Local-to-global node mapping
            dof_i = ID[node_i]
            if dof_i == -1:  # Skip Dirichlet conditions
                continue
            F[dof_i] += fe[i]  # Assemble global force vector
            for j, node_j in enumerate(IEN[e]):
                dof_j = ID[node_j]
                if dof_j == -1:  # Skip Dirichlet conditions
                    continue
                k_row_idx.append(dof_i)
                k_col_idx.append(dof_j)
                k_data.append(ke[i, j])  # Assemble global stiffness matrix
      
    K=scipy.sparse.coo_matrix((k_data,(k_row_idx,k_col_idx)),shape=(n,n)).tocsr()

    # Solve the system of equations for unknowns, d
    d=scipy.sparse.linalg.spsolve(K,F)
    d_tot=ExtractTotalD(ID,d,boundaries)
    # Use ExtractTotalD to identify coefficients with the
    # basis functions indexed as in the basis mesh

    return d_tot




# Determine the solution evaluated at element, e,
#    with Lagrange polynomial basis, basis,
#    and coefficient vector, d_total,
#    and IEN array, IEN,
#    at the point xi_pt
def EvaluateSolution(e, basis, d_total, IEN, xi_pt):
    uhval=0
    for a in range(basis.NBasisFuncs()):
        A=IEN[e][a]
        dA=d_total[A]
        uhval+=dA*basis.EvalBasisFunction(a,xi_pt)
    
    return uhval




# Determine the gradient of the solution 
#    evaluated at element, e,
#    with Lagrange polynomial basis, basis,
#    and coefficient vector, d_total,
#    and IEN array, IEN,
#    and deformation gradient, DF, evaluated at xi_pt
#    at the point xi_pt
def EvaluateSolutionGradient(e, basis, d_total, IEN, DF, xi_pt):
    uhval_grad=np.zeros(len(basis.degs))
    for a in range(basis.NBasisFuncs()):
        A=IEN[e][a]
        dA=d_total[A]
        parametricGradient=basis.EvaluateBasisParametricGradient(a,xi_pt)
        DF_minus=np.linalg.inv(DF)
        DF_minus_transpose=DF_minus.transpose()
        uhval_grad +=dA*DF_minus_transpose@parametricGradient

    return uhval_grad
    
# Input a basis mesh and a list of boundaries, output
# a refined basis mesh where each element has been subdivided
# into 4 and the corresponding boundaries
def RefineAndIterate(bm,boundaries):
    # refine and iterate
    filename="unit_square"
    bm.OutputRefinedOBJFile(filename + ".obj")
    surf_mesh = mesh.SurfaceMesh.FromOBJ_FileName(filename+".obj")
    surf_mesh.AppendSurfaceBoundarySets(filename+"_bndry.obj")
    # mesh.PlotSurfaceMeshWithBoundaryComps(surf_mesh, plot3d=False)

    degx = bm.basis.degs[0]
    degy = bm.basis.degs[1]
    interp_pts_x = bm.basis.interp_pts[0]
    interp_pts_y = bm.basis.interp_pts[1]

    basis = bf.LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)

    new_bm = basismesh.BasisMesh(surf_mesh, basis)
    bndry_data = new_bm.boundaries
    old_boundaries = [bdry for bdry in boundaries]
    new_boundaries = []

    for boundary in bndry_data:
        new_bnd = None
        for old_bdry in old_boundaries:
            if len(boundary.intersection(old_bdry.bdry_nodes)) > 1:
                if old_bdry.type == bnd.BoundaryConditionType.DIRICHLET:
                    new_bnd = bnd.DirichletBoundaryCondition(bnd.BoundaryConditionType.DIRICHLET,boundary, old_bdry.rhs_func)
                elif old_bdry.type == bnd.BoundaryConditionType.NEUMANN:
                    new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN,boundary, old_bdry.rhs_func)
                elif old_bdry.type == bnd.BoundaryConditionType.ROBIN:
                    new_bnd = bnd.RobinBoundaryCondition(bnd.BoundaryConditionType.ROBIN,boundary, old_bdry.u_multiplier, old_bdry.rhs_func)
                else:
                    sys.exit("invalid boundary prescribed")
                break
        new_boundaries.append(new_bnd)


    return new_bm,new_boundaries



# extract the convergence rates given an input basis mesh, bm,
#   input boundary information, boundaries,
#   quadrature points in 2D and 1D, quad_2d and quad_1d, respectively,
#   a matrix-valued thermal conductivity function, kappa,
#   a forcing function, f,
#   a prescribed number of iterations on which to check convergence, num_iterations,
#   the exact solution to the problem, exact_sol,
#   the type of norm used (L2=0, H1=1), normtype,
#   and the gradient of the exact solution, exact_sol_derv,
def ExtractConvergenceRates(bm, boundaries, quad_2d, quad_1d, kappa, f, num_iterations, exact_sol, normtype=0, exact_sol_derv=lambda x, y: 0):
    error_list = []
    h_list = []

    if num_iterations < 2:
        sys.exit("Cannot compute convergence rates on fewer than two data points")
    for i in range(0, num_iterations):     
        # if num_iterations-1: 
        bm,boundaries = RefineAndIterate(bm,boundaries)
        pts = bm.pts
        d_tot=HeatConductionProblem(bm,quad_2d,quad_1d,f,kappa,boundaries)
            
        errorSquared=0
        if normtype==0:
            for e in range(len(bm.IEN)):
                x_pts = [pts[i] for i in bm.IEN[e]]
                for nq in range(len(quad_2d.quad_wts)):
                    Jacobian=bm.basis.EvaluateJacobian(x_pts, *[quad_2d.quad_pts[nq]])
                    points_to_eval=bm.basis.EvaluateSpatialMapping(x_pts, *[quad_2d.quad_pts[nq]])
                    u_exact=exact_sol(points_to_eval[0],points_to_eval[1]) #using evaluatespatialmapping for pts
                    uh = EvaluateSolution(e, bm.basis, d_tot, bm.IEN, quad_2d.quad_pts[nq])
                    u_minus_uh= u_exact-uh
                    errorSquared += (u_minus_uh**2)*quad_2d.quad_wts[nq]*Jacobian

        elif normtype==1:
            for e in range(len(bm.IEN)):
                x_pts = [pts[i] for i in bm.IEN[e]]
                for nq in range(len(quad_2d.quad_wts)):
                    Jacobian=bm.basis.EvaluateJacobian(x_pts, *[quad_2d.quad_pts[nq]])
                    points_to_eval=bm.basis.EvaluateSpatialMapping(x_pts, *[quad_2d.quad_pts[nq]])
                    u_exact=exact_sol(points_to_eval[0],points_to_eval[1]) #using evaluatespatialmapping for pts
                    uh = EvaluateSolution(e, bm.basis, d_tot, bm.IEN, quad_2d.quad_pts[nq])
                    u_minus_uh=u_exact-uh
                    du_dx=exact_sol_derv(points_to_eval[0],points_to_eval[1])
                    DF=bm.basis.EvaluateDeformationGradient(x_pts, *[quad_2d.quad_pts[nq]])
                    duh_dx=EvaluateSolutionGradient(e, bm.basis, d_tot, bm.IEN, DF, *[quad_2d.quad_pts[nq]])
                    du_dx_minus_duh_dx=du_dx-duh_dx
                    grad_norm=np.linalg.norm(du_dx_minus_duh_dx)                   
                    errorSquared += (u_minus_uh**2+grad_norm**2)*quad_2d.quad_wts[nq]*Jacobian
        
        error_list.append(errorSquared)
        h_list.append(np.sqrt(max(bm.mesh.face_areas)))

            

    log_h = [np.log(val) for val in h_list]
    log_e = np.array([np.log(val) for val in error_list])
    log_e *= 0.5  # multiply by 1/2 to account for square root on the norm

    plt.plot(log_h, log_e)
    plt.xlabel("log h")
    plt.ylabel("log e")

    slope = (log_e[-1]-log_e[-2])/(log_h[-1]-log_h[-2])
    print("The convergence rate is ", abs(slope))
    m = 1  # number of derivatives on the weak form
    expect = min(min(bm.basis.degs)+1-normtype, 2*(min(bm.basis.degs)+1-m))
    print("The expected convergence rate is ", expect)

    return log_e, log_h



##################
# Thermal conductivity
def Kappa_block(x, y):
    return np.eye(2)


def Init_Convergence(deg,num_iterations,norm):
    exact_sol= lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)
    exact_sol_derv= lambda x,y: np.array([np.pi*np.cos(np.pi*x)*np.sin(np.pi*y),np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)])

    num_iterations=3

    mesh_obj = "unit_square.obj" #input mesh file
    bdry_obj = "unit_square_boundaries.obj" #input boundary data

    surf_mesh = mesh.SurfaceMesh.FromOBJ_FileName(mesh_obj)
    # Alternatively, enter brick_boundaries.obj into AppendSurfaceBoundarySets
    mesh_bndry_data = surf_mesh.AppendSurfaceBoundarySets(bdry_obj)

    # define input basis functions
    degx = deg
    degy = deg
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
    basis = bf.LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)

    # combine the basis functions with the input geometry
    bm = basismesh.BasisMesh(surf_mesh,basis)
    # extract out new boundary data
    bndry_data = bm.boundaries


    # Define quadrature routines
    n_quad = 2 * max(degx,degy)
    gauss = quad.GaussQuadratureQuadrilateral(n_quad,-1,1)
    gauss_1d = quad.GaussQuadrature1D(n_quad,-1,1)

    # define input force functions, boundary conditions,
    # and other parameters
    g = lambda x,y: 0
    h_right = lambda x,y: -np.pi*np.sin(np.pi*y)
    h_top= lambda x,y: -np.pi*np.sin(np.pi*x)
    f= lambda x,y: 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
    kappa = Kappa_block                                     

    # assign boundary conditions to various boundaries;
    # this particular code assigns each boundary condition to a
    # different boundary component (arbitrarily)
    boundaries = []
    count = 0


    print(len(bndry_data)) ## We have three but we have 4 different types of
    for boundary in bndry_data: ## 0 IS TOP, 1 IS RIGH, 2 IS BOTTOM, 3 IS LEFT
        if count == 0:
            new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN, boundary,h_top)

        elif count == 1:
            new_bnd = bnd.NeumannBoundaryCondition(bnd.BoundaryConditionType.NEUMANN, boundary,h_right)


        elif count == 2:
            new_bnd = bnd.DirichletBoundaryCondition(bnd.BoundaryConditionType.DIRICHLET, boundary, g)

        
        elif count == 3:
            new_bnd = bnd.DirichletBoundaryCondition(bnd.BoundaryConditionType.DIRICHLET, boundary, g)

        count += 1
        boundaries.append(new_bnd)


    ExtractConvergenceRates(bm, boundaries, gauss, gauss_1d, kappa, f, num_iterations, exact_sol, norm, exact_sol_derv)


#### PARAMETERS
bilinear=1
biquadratic=2
iterations=3
L2=0
H1=1

######## CONVERGENCE ##########

########## ---> READ ME <--- #########
#### To run each of these simulations. It's needed to delete unit_square.obj
#### and unit_square_boundaries.obj after every simulation. Otherwise there will be errors
#### or take a long time since files have been refined. This applies also to TestingProblem_block()

# ### L2 Bilinear ###
# Init_Convergence(bilinear,iterations,L2)
# plt.title(r"$L^2$ Convergence Rate for Bilinear")
# plt.show()

# ### H1 Bilinear ###
# Init_Convergence(bilinear,iterations,H1)
# plt.title(r"$H^1$ Convergence Rate for Bilinear")
# plt.show()

### L2 Biquadratic ###
# Init_Convergence(biquadratic,iterations,L2)
# plt.title(r"$L^2$ Convergence Rate for Biquadratic")
# plt.show()

# ### H1 Biquadratic ###
# Init_Convergence(biquadratic,iterations,H1)
# plt.title(r"$H^1$ Convergence Rate for Biquadratic")
# plt.show()

# TestingProblem_block()


# ######## TASK 3 #########
# ## BILINEAR
Brick_Simulation(bilinear)

# ## BIQUADRATIC
# Brick_Simulation(biquadratic)