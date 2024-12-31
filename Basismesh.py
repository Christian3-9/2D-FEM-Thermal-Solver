#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:51:08 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

import Surfacemesh as sm
import MultidimensionalSpatialParametricGradient_Solutions as bs

class BasisMesh():
    
    def __init__(self, surfacemesh,basis):
        self.mesh = surfacemesh
        self.basis = basis
        self.pts = []
        self.IEN = []
        self.boundaries = []
        self.boundaries_lists = []
        
        self.__DefineGlobalBasisFunctions__()
        self.__ModifyBoundaryData__()
        
    def __DefineGlobalBasisFunctions__(self):
        if self.basis.degs[0] != self.basis.degs[1]:
          sys.exit("Cannot currently operate on bases of mixed polynomial degree")
        
        # define coordinates on nodes---only allow two dimensions
        node_coords = self.mesh.vs[:,:2]
        node_2_idx = {}
        counter = 0
        for i in range(0,len(node_coords)):
            node_2_idx[i] = counter
            counter += 1
        
        # iterate through all edges and make coordinates on the edges
        p = self.basis.degs[0]
        edge_coords = {}
        edge_2_idxs = {}
        mesh_edges = self.mesh.get_edges()
        for edge in mesh_edges:
            # subdivide evenly between the start and the end nodes
            start_node = node_coords[edge[0]]
            end_node = node_coords[edge[1]]
            x_coords = np.linspace(start_node[0],end_node[0],p+1)
            y_coords = np.linspace(start_node[1],end_node[1],p+1)
            coords = [np.array((x_coords[i],y_coords[i])) for i in range(1,p)]
            edge_coords[edge] = coords
            edge_idxs = []
            for val in coords:
                edge_idxs.append(counter)
                counter += 1
            edge_2_idxs[edge] = edge_idxs
            
        # iterate through all faces and make coordinates on the faces
        face_coords = {}
        face_2_idxs = {}
        N_1d_0 = lambda xi: 0.5 - 0.5*xi
        N_1d_1 = lambda xi: 0.5 + 0.5*xi
        total_samples = np.linspace(-1,1,p+1)
        interior_samples = total_samples[1:p]
        for idx in range(0,len(self.mesh.faces)):
            if self.mesh.is_quadrilateral_face(idx) == False:
                sys.exit("cannot work on non-quadrilateral faces")
            face = self.mesh.faces[idx]
            face_vs = [node_coords[i] for i in face]
            face_vals = []
            # use Lagrange polynomials of degree 1 to solve positions
            # xi fastest and then eta
            for etaval in interior_samples:
                N0eta = N_1d_0(etaval)
                N1eta = N_1d_1(etaval)
                for xival in interior_samples:
                    N0xi = N_1d_0(xival)
                    N1xi = N_1d_1(xival)
                    # face vertices are in CCW order
                    xval = N0eta*N0xi*face_vs[0]
                    xval += N0eta*N1xi*face_vs[1]
                    xval += N1eta*N1xi*face_vs[2]
                    xval += N1eta*N0xi*face_vs[3]
                    face_vals.append(xval)
            face_coords[idx] = face_vals
            face_idxs = []
            for val in face_vals:
                face_idxs.append(counter)
                counter += 1
            face_2_idxs[idx] = face_idxs

        
        
        # assign a list of global points associated with basis
        # functions that first loops through mesh vertices, then
        # mesh edges, and finally mesh faces
        x_coords = []
        for node in node_coords:
            x_coords.append(node) # append all mesh vertices
        
        for edge in mesh_edges:
            edge_pts = edge_coords[edge]
            for pt in edge_pts:
                x_coords.append(pt)
                
        for idx in range(0,len(self.mesh.faces)):
            face_pts = face_coords[idx]
            for pt in face_pts:
                x_coords.append(pt)
        
        
        
        
        
        face_IEN = []
        # arrange all degrees of freedom on a face to go in
        # xi-fastest and then eta order
        for idx in range(0,len(self.mesh.faces)):
            this_IEN = []
            
            face = self.mesh.faces[idx]
            # find the halfedge of the face whose FromVertex
            # is the first vertex of the face
            hes = self.mesh.get_face_halfedges(idx)
            init_he = self.mesh.FaceInitVertexFromHalfEdge(idx)
            for he in hes:
                if self.mesh.FromVertex(he) == face[0]:
                    init_he = he
            if init_he == None:
                sys.exit("Topology of the mesh is incorrect---no halfedge of this vertex owns the first vertex of the face")
            
            next_he = self.mesh.NextHE(init_he)
            prev_he = self.mesh.PrevHE(init_he)
            other_he = self.mesh.NextHE(next_he)
            
            # iterate on the bottom edge
            # get the initial vertex
            this_IEN.append(node_2_idx[face[0]])
            # get the edge points along init_he
            init_edge = mesh_edges[init_he.edge]
            init_edge_idxs = edge_2_idxs[init_edge]
            if init_edge[0] == face[0]: # orientation is the same
                for i in range(0,len(init_edge_idxs)):
                    this_IEN.append(init_edge_idxs[i])
            else: # orientation is reversed
                for i in range(len(init_edge_idxs)-1,-1,-1):
                    this_IEN.append(init_edge_idxs[i])
            this_IEN.append(node_2_idx[face[1]])
            
            # now iterate over the sides and the face
            prev_edge = mesh_edges[prev_he.edge]
            prev_edge_idxs = edge_2_idxs[prev_edge]
            next_edge = mesh_edges[next_he.edge]
            next_edge_idxs = edge_2_idxs[next_edge]

            left_edge_idxs = [i for i in prev_edge_idxs]
            # left edge is oriented in the same way as the half-edge,
            # and thus the order must be flipped to be correct
            if prev_edge[1] == face[0]: 
                for i in range(0,len(prev_edge_idxs)):
                    left_edge_idxs[i] = prev_edge_idxs[len(prev_edge_idxs)-1-i]
            
            right_edge_idxs = [i for i in next_edge_idxs]
            # right edge is oriented in the opposite way as the half-edge,
            # and thus the order must be flipped to be correct
            if next_edge[1] == face[1]:
                for i in range(0,len(next_edge_idxs)):
                    right_edge_idxs[i] = next_edge_idxs[len(next_edge_idxs)-1-i]

            # iterate sides through the interior
            face_idxs = face_2_idxs[idx]
            counter = 0
            for i in range(0,p-1):
                this_IEN.append(left_edge_idxs[i])
                for j in range(0,p-1):
                    this_IEN.append(face_idxs[counter])
                    counter += 1
                this_IEN.append(right_edge_idxs[i])
                
            # iterate over the top edge
            this_IEN.append(node_2_idx[face[3]])
            
            other_edge = mesh_edges[other_he.edge]
            other_edge_idxs = edge_2_idxs[other_edge]
            top_edge_idxs = [i for i in other_edge_idxs]
            # top edge is oriented in the same way as the half-edge,
            # and thus the order must be flipped to be correct
            if other_edge[1] == face[3]: 
                for i in range(0,len(other_edge_idxs)):
                    top_edge_idxs[i] = other_edge_idxs[len(other_edge_idxs)-1-i]

            for idx in top_edge_idxs:
                this_IEN.append(idx)
            
            this_IEN.append(node_2_idx[face[2]])
            
            face_IEN.append(this_IEN)
        self.IEN = face_IEN
        self.pts = x_coords
        
    # include all the new points in the boundary degrees of freedom
    def __ModifyBoundaryData__(self):
        mesh_bdries = self.mesh.boundary_set_vert_idxs
        # mesh boundary indices should match basismesh indices
        new_bdries = []
        bdry_counter = 0
        for bdry in  mesh_bdries:
            first_vert = self.mesh.first_boundary_vert_idxs[bdry_counter][0]
            # find the first vertex
            hes = self.mesh.get_vertex_halfedges(first_vert)
            counter = 0
            current_he = None
            for he in hes:
                if he.ToVertex() == self.mesh.first_boundary_vert_idxs[bdry_counter][1]:
                    current_he = he
                    break
                else:
                    counter += 1
            
            init_he = current_he
            
            # input first vertex first
            new_bdry = [first_vert]
            while True:
                # determine the face edge that the halfedge is on
                if current_he.face == -1:
                    he_face_idx = self.mesh.OppositeHE(current_he).face
                else:
                    he_face_idx = current_he.face
                current_face = self.mesh.faces[he_face_idx]
                
                # orient the halfedge relative to the face
                from_vert = self.mesh.FromVertex(current_he)
                to_vert = self.mesh.ToVertex(current_he)
                p = self.basis.degs[0]
                # bottom of face
                if from_vert == current_face[0] and to_vert == current_face[1]:
                    for i in range(1,p+1):
                        new_bdry.append(self.IEN[he_face_idx][i])
                elif from_vert == current_face[1] and to_vert == current_face[0]:
                    for i in range(1,p+1):
                        new_bdry.append(self.IEN[he_face_idx][p - i])
                # right of face
                elif from_vert == current_face[1] and to_vert == current_face[2]:
                    for i in range(1,p+1):
                        new_bdry.append(self.IEN[he_face_idx][(p+1)*(i+1)-1])
                elif from_vert == current_face[2] and to_vert == current_face[1]:
                    for i in range(1,p+1):
                        new_bdry.append(self.IEN[he_face_idx][(p+1)**2-1-(p+1)*(i)])
                # top of face
                elif from_vert == current_face[2] and to_vert == current_face[3]:
                    for i in range(1,p+1):
                        new_bdry.append(self.IEN[he_face_idx][(p+1)**2-1-i])
                elif from_vert == current_face[3] and to_vert == current_face[2]:
                    for i in range(1,p+1):
                        new_bdry.append(self.IEN[he_face_idx][(p+1)*p+i])
                # right of face
                elif from_vert == current_face[3] and to_vert == current_face[0]:
                    for i in range(1,p+1):
                        new_bdry.append(self.IEN[he_face_idx][(p+1)*(p-i)])
                elif from_vert == current_face[0] and to_vert == current_face[3]:
                    for i in range(1,p+1):
                        new_bdry.append(self.IEN[he_face_idx][(p+1)*i])


                
                
                if current_he.face == -1: # on boundary
                    current_he = self.mesh.NextHE(current_he)
                else:
                    opp_he = self.mesh.OppositeHE(current_he)
                    prev_he = self.mesh.PrevHE(opp_he)
                    current_he = self.mesh.OppositeHE(prev_he)
                
                if init_he == current_he:
                    break
                elif self.mesh.ToVertex(current_he) not in bdry \
                    or self.mesh.FromVertex(current_he) not in bdry:
                        break
                
            new_bdries.append(set(new_bdry))
            self.boundaries_lists.append(new_bdry)
            bdry_counter += 1
        self.boundaries = new_bdries
        
        
    # output an OBJ file that is subdivided
    def OutputRefinedOBJFile(self,filename):
        # create a new BasisMesh where you assume that it is of degree 2
        degx = 2
        degy = 2
        interp_pts_x = np.linspace(-1,1,degx+1)
        interp_pts_y = np.linspace(-1,1,degy+1)
        pretend_basis = bs.LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)

        bm = BasisMesh(self.mesh,pretend_basis)
        
        # subdivision has now occurred on the basis mesh;
        # output new vertices and faces
        with open(filename,'w') as file:
            for v in bm.pts:
                file.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(0) + "\n")
                
            for f in bm.IEN:
                # do the bottom-left "face"
                file.write("f " + str(f[0]+1) + " " + str(f[1]+1) + " " + str(f[4]+1) + " " + str(f[3]+1) + "\n")
                # do the bottom-right "face"
                file.write("f " + str(f[1]+1) + " " + str(f[2]+1) + " " + str(f[5]+1) + " " + str(f[4]+1) + "\n")
                # do the top-left "face"
                file.write("f " + str(f[3]+1) + " " + str(f[4]+1) + " " + str(f[7]+1) + " " + str(f[6]+1) + "\n")
                # do the top-right "face"
                file.write("f " + str(f[4]+1) + " " + str(f[5]+1) + " " + str(f[8]+1) + " " + str(f[7]+1) + "\n")
    
        splitname = filename.split('.obj')
        counter = 1
        with open(splitname[0] + "_bndry.obj", 'w') as file:
            for boundary in bm.boundaries_lists:
                # output points
                for i in boundary:
                    pt = bm.pts[i]
                    file.write("v " + str(pt[0]) + " " + str(pt[1]) + " " + str(0) + "\n")
                
                # output boundary data
                file.write("l")
                for i in boundary:
                    file.write(" " + str(counter))
                    counter += 1
                file.write("\n")
    
        
def TestTempMesh():
    mesh=sm.SurfaceMesh.FromOBJ_FileName("temp_mesh.obj")
    mesh.AppendSurfaceBoundarySets('')
    face_areas = []
    for i in range(0,len(mesh.faces)):
        face_areas.append(mesh.GetFaceArea(i))
    
    # sm.PlotSurfaceMesh(mesh,face_areas)
    # sm.PlotSurfaceMeshIndexing(mesh)
    
    degx = 2
    degy = 2
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
    basis = bs.LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    bm = BasisMesh(mesh,basis)
    
    # print(bm.IEN)
    # print(bm.pts)
    
    bm.OutputRefinedOBJFile("temp_mesh_refined.obj")
    mesh_ref = sm.SurfaceMesh.FromOBJ_FileName("temp_mesh_refined.obj")
    sm.PlotSurfaceMeshIndexing(mesh_ref)

    # combine the basis functions with the input geometry
    bm = BasisMesh(mesh_ref,basis)
    # extract out new boundary data
    bndry_data = bm.boundaries
    plt.show()
    
# TestTempMesh()