#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Yotam Gingold <yotam (strudel) yotamgingold.com>
License: Public Domain.  (I, Yotam Gingold, the author, release this code into the public domain.)
GitHub: https://github.com/yig/trimesh

surfacemesh.py:
    A python half-edge data structure for a mixed triangle/quadrilateral mesh that can
        - handle boundaries (the first outgoing halfedge at a vertex is always a boundary halfedge)
          (and the last outgoing halfedge is always the opposite of a boundary halfedge))
        - save/load itself to/from OBJ format
        - save itself to OFF format

    Loading:
        import surfacemesh
        mesh = surfacemesh.SurfaceMesh.FromOBJ_FileName( "cube.obj" )
"""

from __future__ import print_function, division
from numpy import *
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import matplotlib

import sys

def mag2( vec ):
    return dot( vec, vec )
def mag( vec ):
    return sqrt(mag2(vec))

class SurfaceMesh( object ):
    def __init__( self ):
        self.vs = []
        self.faces = []
        
        self.is_all_tri_mesh = False
        self.is_all_quad_mesh = False
        
        self.__face_normals = None
        self.__face_areas = None
        self.__face_to_halfedges = None
        self.__vertex_normals = None
        self.__vertex_areas = None
        self.__edges = None
        self.__face_centroid = None
        
        self.__halfedges = None
        self.__vertex_halfedges = None
        self.__face_halfedges = None
        self.__edge_halfedges = None
        self.__directed_edge2he_index = None
        
        self.lifetime_counter = 0
        self.used_default_boundaries = True
        self.boundary_set_vert_idxs = []
        self.first_boundary_vert_idxs = []
    
    def is_triangle_face( self, i ):
        return self.faces[i][3] < 0
    
    def is_quadrilateral_face( self, i ):
        return self.faces[i][3] >= 0
    
    def copy( self ):
        import copy
        return copy.deepcopy( self )
    
    def __deepcopy__( self, memodict ):
        result = SurfaceMesh()
        
        ## Make a copy of vs and faces using array().
        ## But in case they weren't stored as arrays, return them as the type they were.
        ## This is important if they were lists, and someone expected to be able to call
        ## .append() or .extend() on them.
        result.vs = array( self.vs )
        if not isinstance( self.vs, ndarray ):
            result.vs = type( self.vs )( result.vs )
        
        result.faces = array( self.faces )
        if not isinstance( self.faces, ndarray ):
            result.faces = type( self.faces )( result.faces )
        
        if hasattr( self, 'uvs' ):
            result.uvs = array( self.uvs )
            if not isinstance( self.uvs, ndarray ):
                result.uvs = type( self.uvs )( result.uvs )
        
        ## I could skip copying these cached values, but they are usually needed for rendering
        ## and copy quickly.
        if self.__face_normals is not None:
            result.__face_normals = self.__face_normals.copy()
        if self.__face_areas is not None:
            result.__face_areas = self.__face_areas.copy()
        if self.__vertex_normals is not None:
            result.__vertex_normals = self.__vertex_normals.copy()
        if self.__vertex_areas is not None:
            result.__vertex_areas = self.__vertex_areas.copy()
        if self.__edges is not None:
            result.__edges = list( self.__edges )
        
        ## I will skip copying these cached values, because they copy slowly and are
        ## not as commonly needed.  They'll still be regenerated as needed.
        '''
        if self.__halfedges is not None:
            from copy import copy
            result.__halfedges = [ copy( he ) for he in self.__halfedges ]
        if self.__vertex_halfedges is not None:
            result.__vertex_halfedges = list( self.__vertex_halfedges )
        if self.__face_halfedges is not None:
            result.__face_halfedges = list( self.__face_halfedges )
        if self.__edge_halfedges is not None:
            result.__edge_halfedges = list( self.__edge_halfedges )
        if self.__directed_edge2he_index is not None:
            result.__directed_edge2he_index = dict( self.__directed_edge2he_index )
        '''
        
        result.lifetime_counter = self.lifetime_counter
        
        return result
    
    # TODO Kendrick: Fix this for quads vs triangles
    def update_face_normals_and_areas( self ):
        if self.__face_normals is None: self.__face_normals = zeros( ( len( self.faces ), 3 ) )
        if self.__face_areas is None: self.__face_areas = zeros( len( self.faces ) )
        
        ## We need subtraction between vertices.
        ## Convert vertices to arrays once here, or else we'd have to call asarray()
        ## ~6 times for each vertex.
        ## NOTE: If self.vs is already an array, then this code does nothing.
        ## TODO Q: Should I set self.vs = asarray( self.vs )?  It might violate someone's
        ##         assumption that self.vs is whatever indexable type they left it.
        ##         In particular, this violates the ability of someone to .append() or .extend()
        ##         self.vs.
        vs = asarray( self.vs )
        fs = asarray( self.faces, dtype = int )
        
        ## Slow:
        '''
        for f in xrange( len( self.faces ) ):
            face = self.faces[f]
            n = cross(
                vs[ face[1] ] - vs[ face[0] ],
                vs[ face[2] ] - vs[ face[1] ]
                )
            nmag = mag( n )
            self.__face_normals[f] = (1./nmag) * n
            self.__face_areas[f] = .5 * nmag
        '''
        ## ~Slow
        
        ## Fast:
        self.__face_normals = cross( vs[ fs[:,1] ] - vs[ fs[:,0] ], vs[ fs[:,2] ] - vs[ fs[:,1] ] )
        self.__face_areas = sqrt((self.__face_normals**2).sum(axis=1))
        self.__face_normals /= self.__face_areas[:,newaxis]
        self.__face_areas *= 0.5
        ## ~Fast
        
        assert len( self.faces ) == len( self.__face_normals )
        assert len( self.faces ) == len( self.__face_areas )
    
    
    def get_face_normals( self ):
        if self.__face_normals is None: self.update_face_normals_and_areas()
        return self.__face_normals
    
    face_normals = property( get_face_normals )
    
    
    def get_face_areas( self ):
        if self.__face_areas is None: self.update_face_normals_and_areas()
        return self.__face_areas
    
    face_areas = property( get_face_areas )
    
    
    def update_vertex_normals( self ):
        if self.__vertex_normals is None: self.__vertex_normals = zeros( ( len(self.vs), 3 ) )
        
        ## Slow:
        '''
        for vi in xrange( len( self.vs ) ):
            self.__vertex_normals[vi] = 0.
            
            for fi in self.vertex_face_neighbors( vi ):
                ## This matches the OpenMesh FAST vertex normals.
                #self.__vertex_normals[vi] += self.face_normals[ fi ]
                ## Area weighted
                self.__vertex_normals[vi] += self.face_normals[ fi ] * self.face_areas[ fi ]
        
        ## Now normalize the normals
        #self.__vertex_normals[vi] *= 1./mag( self.__vertex_normals[vi] )
        self.__vertex_normals *= 1./sqrt( ( self.__vertex_normals**2 ).sum(1) ).reshape( (len(self.vs), 1) )
        '''
        ## ~Slow
        
        ## Fast:
        fs = asarray( self.faces, dtype = int )
        ## This matches the OpenMesh FAST vertex normals.
        #fns = self.face_normals
        ## Area weighted
        fns = self.face_normals * self.face_areas[:,newaxis]
        
        self.__vertex_normals[:] = 0.
        ## I wish this worked, but it doesn't do the right thing with aliasing
        ## (when the same element appears multiple times in the slice).
        #self.__vertex_normals[ fs[:,0] ] += fns
        #self.__vertex_normals[ fs[:,1] ] += fns
        #self.__vertex_normals[ fs[:,2] ] += fns
        import itertools
        for c in (0,1,2):
            for i, n in itertools.izip( fs[:,c], fns ):
                self.__vertex_normals[ i ] += n
        
        self.__vertex_normals /= sqrt( ( self.__vertex_normals**2 ).sum(axis=1) )[:,newaxis]
        ## ~Fast
        
        assert len( self.vs ) == len( self.__vertex_normals )
    
    def get_vertex_normals( self ):
        if self.__vertex_normals is None: self.update_vertex_normals()
        return self.__vertex_normals
    
    vertex_normals = property( get_vertex_normals )
    
    
    def update_vertex_areas( self ):
        if self.__vertex_areas is None: self.__vertex_areas = zeros( len(self.vs) )
        
        ## Slow:
        '''
        for vi in xrange( len( self.vs ) ):
            ## Try to compute proper area (if we have laplacian editing around).
            ## (This only matters for obtuse triangles.)
            try:
                #raise ImportError
                import laplacian_editing
                cot_alpha, cot_beta, area = laplacian_editing.cotangentWeights(
                    self.vs[ vi ],
                    [ self.vs[ vni ] for vni in self.vertex_vertex_neighbors( vi ) ],
                    self.vertex_is_boundary( vi )
                    )
                self.__vertex_areas[vi] = area
            
            ## Otherwise use 1/3 of the incident faces' areas
            except ImportError:
                self.__vertex_areas[vi] = 0.
                for fi in self.vertex_face_neighbors( vi ):
                    self.__vertex_areas[vi] += self.face_areas[ fi ]
                
                self.__vertex_areas[vi] *= 1./3.
        '''
        ## ~Slow
        
        ## Fast:
        ## NOTE: This does not use laplacian_editing's so-called mixed area
        ##       computation even if the module is present!
        ##       (This only matters for obtuse triangles.)
        self.__vertex_areas[:] = 0.
        
        fs = asarray( self.faces, dtype = int )
        fas = self.__face_areas
        ## I wish this worked, but it doesn't do the right thing with aliasing
        ## (when the same element appears multiple times in the slice).
        #self.__vertex_areas[ fs[:,0] ] += fas
        #self.__vertex_areas[ fs[:,1] ] += fas
        #self.__vertex_areas[ fs[:,2] ] += fas
        import itertools
        for c in (0,1,2):
            for i, area in itertools.izip( fs[:,c], fas ):
                self.__vertex_areas[ i ] += area
        
        self.__vertex_areas /= 3.
        ## ~Fast
        
        assert len( self.vs ) == len( self.__vertex_areas )
    
    def get_vertex_areas( self ):
        if self.__vertex_areas is None: self.update_vertex_areas()
        return self.__vertex_areas
    
    vertex_areas = property( get_vertex_areas )
    
    
    def update_edge_list( self ):
        #from sets import Set, ImmutableSet
        Set, ImmutableSet = set, frozenset
        
        ## We need a set of set-pairs of vertices, because edges are bidirectional.
        edges = Set()
        for face in self.faces:
            if face[3]<0:
                edges.add( ImmutableSet( ( face[0], face[1] ) ) )
                edges.add( ImmutableSet( ( face[1], face[2] ) ) )
                edges.add( ImmutableSet( ( face[2], face[0] ) ) )
            else:
                edges.add( ImmutableSet( ( face[0], face[1] ) ) )
                edges.add( ImmutableSet( ( face[1], face[2] ) ) )
                edges.add( ImmutableSet( ( face[2], face[3] ) ) )
                edges.add( ImmutableSet( ( face[3], face[0] ) ) )
        
        self.__edges = [ tuple( edge ) for edge in edges ]
    
    def get_edges( self ):
        if self.__edges is None: self.update_edge_list()
        return self.__edges
    
    edges = property( get_edges )
    
    
    class HalfEdge( object ):
        def __init__( self ):
            self.to_vertex = -1
            self.face = -1
            self.edge = -1
            self.opposite_he = -1
            self.next_he = -1
        
        def ToVertex(self):
            return self.to_vertex
        
    def NextHE( self, he ):
        next_he = he.next_he
        if self.__halfedges is None: self.update_halfedges()
        return self.__halfedges[next_he]

    def OppositeHE( self, he ):
        opp_he = he.opposite_he
        if self.__halfedges is None: self.update_halfedges()
        return self.__halfedges[opp_he]

    def CurrentHEIdx( self, he ):
        opp_he = self.OppositeHE(he)
        return opp_he.opposite_he
    
    def HEFromIdx( self, he_idx ):
        return self.__halfedges[he_idx]

    def PrevHE( self, he ):
        he_idx = self.CurrentHEIdx(he)
        current_he = self.NextHE(he)
        while current_he.next_he != he_idx:
            current_he = self.NextHE(current_he)
        return self.__halfedges[self.CurrentHEIdx(current_he)]
    
    def ToVertex( self, he ):
        return he.to_vertex
    def FromVertex( self, he ):
        return self.OppositeHE(he).to_vertex
    
    def HalfEdgeToVector( self, he ):
        from_vert = self.FromVertex(he)
        to_vert = self.ToVertex(he)
        
        return self.TwoVertsToVector( from_vert, to_vert )
        
    def TwoVertsToVector( self, from_vert, to_vert ):
        from_loc = self.vs[from_vert]
        to_loc = self.vs[to_vert]
        
        return to_loc - from_loc


    def FaceInitVertexFromHalfEdge( self, face_id ):
        current_he = self.__face_to_halfedges[face_id]
        vert_idx = self.FromVertex(current_he)
        if vert_idx == self.faces[face_id][0]:
            return current_he
        else:
            current_he = self.NextHE(current_he)
            while self.FromVertex(current_he) != vert_idx:
                current_he = self.NextHE(current_he)
            return current_he
        
    def GetFacesAdjacentToVertex( self, vertex_id ):
        ## It's important to access self.halfedges first (which calls get_halfedges()),
        ## so that we're sure all halfedge info is generated.
        halfedges = self.halfedges
        
        he_idx = self.get_vertex_halfedge(vertex_id)
        vert_he = self.__halfedges[he_idx]
        iter_he = vert_he
        adj_faces = []
        
        while True:
            # Store the face if it is not on the boudary
            if iter_he.face != -1:
                adj_faces.append(iter_he.face)
            
            iter_he = self.OppositeHE(iter_he)
            iter_he = self.NextHE(iter_he)
            
            if iter_he == vert_he:
                break
        
        return adj_faces
        
        
    def get_face_halfedge( self, face_id ):
        if self.__face_to_halfedges is None:
            self.__face_to_halfedges = [ None for face in self.faces ]
            for he in self.halfedges:
                if he.face == -1: # not on a face
                    continue
                elif self.__face_to_halfedges[he.face] is None:
                    current_he = he
                    vert_idx = self.faces[he.face][0]
                    if vert_idx == self.FromVertex(current_he):
                        self.__face_to_halfedges[he.face] = current_he
                    else:
                        current_he = self.NextHE(current_he)
                        while self.FromVertex(current_he) != vert_idx:
                            current_he = self.NextHE(current_he)
                        self.__face_to_halfedges[he.face] = current_he
                else:
                    continue
        return self.__face_to_halfedges[face_id]
    
    def get_face_halfedges( self, face_id ):
        face_hes = [self.get_face_halfedge(face_id)]
        next_he = self.NextHE(face_hes[0])

        while face_hes[0] != next_he:
            face_hes.append(next_he)
            next_he = self.NextHE(next_he)
            
        return face_hes
    
    def get_face_edges( self, face_id ):
        face_hes = self.get_face_halfedges(face_id)
        edges = []
        for he in face_hes:
            he_idx = self.CurrentHEIdx(he)
            edges.append(self.he_index2directed_edge(he_idx))
        return edges

    
    def get_face_interior_edges( self, face_id ):
        face_hes = self.get_face_halfedges(face_id)
        interior_edges = []
        for he in face_hes:
            if self.OppositeHE(he).face < 0:
                continue
            else:
                he_idx = self.CurrentHEIdx(he)
                interior_edges.append(self.he_index2directed_edge(he_idx))
        return interior_edges
    
    def get_face_boundary_edges( self, face_id ):
        face_hes = self.get_face_halfedges(face_id)
        boundary_edges = []
        for he in face_hes:
            if self.OppositeHE(he).face < 0:
                he_idx = self.CurrentHEIdx(he)
                boundary_edges.append(self.he_index2directed_edge(he_idx))
            else:
                continue
        return boundary_edges

        
    def get_vertex_halfedge( self, vertex_id ):
        if self.__vertex_halfedges is None:
            self.get_halfedges()
            #sys.exit("Vertex halfedges not yet defined")
        
        return self.__vertex_halfedges[vertex_id]
    
    def get_vertex_halfedges( self, vertex_id ):
        halfedges = self.halfedges
        result = []
        start_he = halfedges[ self.__vertex_halfedges[ vertex_id ] ]
        he = start_he
        while True:
            result.append( he )
            
            he = halfedges[ halfedges[ he.opposite_he ].next_he ]
            if he is start_he: break
        
        return result

        
    def update_halfedges( self ):
        '''
        Generates all half edge data structures for the mesh given by its vertices 'self.vs'
        and faces 'self.faces'.
        
        untested
        '''
        
        self.__halfedges = []
        self.__vertex_halfedges = None
        self.__face_halfedges = None
        self.__edge_halfedges = None
        self.__directed_edge2he_index = {}
        
        __directed_edge2face_index = {}
        for fi, face in enumerate( self.faces ):
            if face[3]<0:
                __directed_edge2face_index[ (face[0], face[1]) ] = fi
                __directed_edge2face_index[ (face[1], face[2]) ] = fi
                __directed_edge2face_index[ (face[2], face[0]) ] = fi
            else:
                __directed_edge2face_index[ (face[0], face[1]) ] = fi
                __directed_edge2face_index[ (face[1], face[2]) ] = fi
                __directed_edge2face_index[ (face[2], face[3]) ] = fi
                __directed_edge2face_index[ (face[3], face[0]) ] = fi
        
        def directed_edge2face_index( edge ):
            result = __directed_edge2face_index.get( edge, -1 )
            
            ## If result is -1, then there's no such face in the mesh.
            ## The edge must be a boundary edge.
            ## In this case, the reverse orientation edge must have a face.
            if -1 == result:
                assert edge[::-1] in __directed_edge2face_index
            
            return result
        
        self.__vertex_halfedges = [None] * len( self.vs )
        self.__face_halfedges = [None] * len( self.faces )
        self.__edge_halfedges = [None] * len( self.edges )
        
        for ei, edge in enumerate( self.edges ):
            he0 = self.HalfEdge()
            ## The face will be -1 if it is a boundary half-edge.
            he0.face = directed_edge2face_index( edge )
            he0.to_vertex = edge[1]
            he0.edge = ei
            
            he1 = self.HalfEdge()
            ## The face will be -1 if it is a boundary half-edge.
            he1.face = directed_edge2face_index( edge[::-1] )
            he1.to_vertex = edge[0]
            he1.edge = ei
            
            ## Add the HalfEdge structures to the list.
            he0index = len( self.__halfedges )
            self.__halfedges.append( he0 )
            he1index = len( self.__halfedges )
            self.__halfedges.append( he1 )
            
            ## Now we can store the opposite half-edge index.
            he0.opposite_he = he1index
            he1.opposite_he = he0index
            
            ## Also store the index in our __directed_edge2he_index map.
            assert edge not in self.__directed_edge2he_index
            assert edge[::-1] not in self.__directed_edge2he_index
            self.__directed_edge2he_index[ edge ] = he0index
            self.__directed_edge2he_index[ edge[::-1] ] = he1index
            
            ## If the vertex pointed to by a half-edge doesn't yet have an out-going
            ## halfedge, store the opposite halfedge.
            ## Also, if the vertex is a boundary vertex, make sure its
            ## out-going halfedge a boundary halfedge.
            ## NOTE: Halfedge data structure can't properly handle butterfly vertices.
            ##       If the mesh has butterfly vertices, there will be multiple outgoing
            ##       boundary halfedges.  Because we have to pick one as the vertex's outgoing
            ##       halfedge, we can't iterate over all neighbors, only a single wing of the
            ##       butterfly.
            if self.__vertex_halfedges[ he0.to_vertex ] is None or -1 == he1.face:
                self.__vertex_halfedges[ he0.to_vertex ] = he0.opposite_he
            if self.__vertex_halfedges[ he1.to_vertex ] is None or -1 == he0.face:
                self.__vertex_halfedges[ he1.to_vertex ] = he1.opposite_he
            
            ## If the face pointed to by a half-edge doesn't yet have a
            ## halfedge pointing to it, store the halfedge.
            if -1 != he0.face and self.__face_halfedges[ he0.face ] is None:
                self.__face_halfedges[ he0.face ] = he0index
            if -1 != he1.face and self.__face_halfedges[ he1.face ] is None:
                self.__face_halfedges[ he1.face ] = he1index
            
            ## Store one of the half-edges for the edge.
            assert self.__edge_halfedges[ ei ] is None
            self.__edge_halfedges[ ei ] = he0index
        
        ## Now that all the half-edges are created, set the remaining next_he field.
        ## We can't yet handle boundary halfedges, so store them for later.
        boundary_heis = []
        for hei, he in enumerate( self.__halfedges ):
            ## Store boundary halfedges for later.
            if -1 == he.face:
                boundary_heis.append( hei )
                continue
            
            face = self.faces[ he.face ]
            i = he.to_vertex
            if face[3]<0:
                j = face[ ( list(face).index( i ) + 1 ) % 3 ]
            else:
                j = face[ ( list(face).index( i ) + 1 ) % 4 ]
            
            he.next_he = self.__directed_edge2he_index[ (i,j) ]
        
        ## Make a map from vertices to boundary halfedges (indices) originating from them.
        ## NOTE: There will only be multiple originating boundary halfedges at butterfly vertices.
        vertex2outgoing_boundary_hei = {}
        #from sets import Set
        Set = set
        for hei in boundary_heis:
            originating_vertex = self.__halfedges[ self.__halfedges[ hei ].opposite_he ].to_vertex
            vertex2outgoing_boundary_hei.setdefault(
                originating_vertex, Set()
                ).add( hei )
            if len( vertex2outgoing_boundary_hei[ originating_vertex ] ) > 1:
                print('Butterfly vertex encountered')
        
        ## For each boundary halfedge, make its next_he one of the boundary halfedges
        ## originating at its to_vertex.
        for hei in boundary_heis:
            he = self.__halfedges[ hei ]
            for outgoing_hei in vertex2outgoing_boundary_hei[ he.to_vertex ]:
                he.next_he = outgoing_hei
                vertex2outgoing_boundary_hei[ he.to_vertex ].remove( outgoing_hei )
                break
        
        assert False not in [ 0 == len( out_heis ) for out_heis in vertex2outgoing_boundary_hei.values() ]
    
    def he_index2directed_edge( self, he_index ):
        '''
        Given the index of a HalfEdge, returns the corresponding directed edge (i,j).
        
        untested
        '''
        
        he = self.halfedges[ he_index ]
        return ( self.halfedges[ he.opposite_he ].to_vertex, he.to_vertex )
    
    def directed_edge2he_index( self, edge ):
        '''
        Given a directed edge (i,j), returns the index of the HalfEdge class in
        halfedges().
        
        untested
        '''
        
        if self.__directed_edge2he_index is None: self.update_halfedges()
        
        edge = tuple( edge )
        return self.__directed_edge2he_index[ edge ]
    
    def get_halfedges( self ):
        '''
        Returns a list of all HalfEdge classes.
        
        untested
        '''
        
        if self.__halfedges is None: self.update_halfedges()
        return self.__halfedges
    
    halfedges = property( get_halfedges )
    
    def vertex_vertex_neighbors( self, vertex_index ):
        '''
        Returns the vertex neighbors (as indices) of the vertex 'vertex_index'.
        
        untested
        '''
        
        ## It's important to access self.halfedges first (which calls get_halfedges()),
        ## so that we're sure all halfedge info is generated.
        halfedges = self.halfedges
        result = []
        start_he = halfedges[ self.__vertex_halfedges[ vertex_index ] ]
        he = start_he
        while True:
            result.append( he.to_vertex )
            
            he = halfedges[ halfedges[ he.opposite_he ].next_he ]
            if he is start_he: break
        
        return result
    
    def vertex_valence( self, vertex_index ):
        '''
        Returns the valence (number of vertex neighbors) of vertex with index 'vertex_index'.
        
        untested
        '''
        
        return len( self.vertex_vertex_neighbors( vertex_index ) )
    
    def vertex_face_neighbors( self, vertex_index ):
        '''
        Returns the face neighbors (as indices) of the vertex 'vertex_index'.
        
        untested
        '''
        
        ## It's important to access self.halfedges first (which calls get_halfedges()),
        ## so that we're sure all halfedge info is generated.
        halfedges = self.halfedges
        result = []
        start_he = halfedges[ self.__vertex_halfedges[ vertex_index ] ]
        he = start_he
        while True:
            if -1 != he.face: result.append( he.face )
            
            he = halfedges[ halfedges[ he.opposite_he ].next_he ]
            if he is start_he: break
        
        return result
    
    def vertex_is_boundary( self, vertex_index ):
        '''
        Returns whether the vertex with given index is on the boundary.
        
        untested
        '''
        
        ## It's important to access self.halfedges first (which calls get_halfedges()),
        ## so that we're sure all halfedge info is generated.
        halfedges = self.halfedges
        return -1 == halfedges[ self.__vertex_halfedges[ vertex_index ] ].face
    
    def edge_is_boundary( self, edge ):
        he_idx = self.directed_edge2he_index(edge)
        he = self.HEFromIdx(he_idx)
        if he.face < 0 or self.OppositeHE(he).face < 0:
            return True
        else:
            return False
    
    def boundary_vertices( self ):
        '''
        Returns a list of the vertex indices on the boundary.
        
        untested
        '''
        
        result = []
        for hei, he in enumerate( self.halfedges ):
            if -1 == he.face:
                # result.extend( self.he_index2directed_edge( hei ) )
                result.append( he.to_vertex )
                result.append( self.halfedges[ he.opposite_he ].to_vertex )
        
        #from sets import ImmutableSet
        ImmutableSet = frozenset
        return list(ImmutableSet( result ))
    
    def boundary_edges( self ):
        '''
        Returns a list of boundary edges (i,j).  If (i,j) is in the result, (j,i) will not be.
        
        untested
        '''
        
        result = []
        for hei, he in enumerate( self.halfedges ):
            if -1 == he.face:
                result.append( self.he_index2directed_edge( hei ) )
        return result
    
    # get all vertices incident to a face
    def face_vertex_neighbors( self, face_id ):
        f_hes = self.get_face_halfedges(face_id)
        verts = []
        for he in f_hes:
            verts.append(self.FromVertex(he))
        return verts
    
    # get all faces sharing an edge with this face
    def face_face_neighbors( self, face_id ):
        f_hes = self.get_face_halfedges(face_id)
        faces = []
        for he in f_hes:
            f = self.OppositeHE(he).face
            if f >= 0:
                faces.append(f)
        return faces

    
    def positions_changed( self ):
        '''
        Notify the object that vertex positions changed.
        All position-related structures (normals, areas) will be marked for re-calculation.
        '''
        
        self.__face_normals = None
        self.__face_areas = None
        self.__vertex_normals = None
        self.__vertex_areas = None
        
        self.lifetime_counter += 1
    
    
    def topology_changed( self ):
        '''
        Notify the object that topology (faces or #vertices) changed.
        All topology-related structures (halfedges, edge lists) as well as position-related
        structures (normals, areas) will be marked for re-calculation.
        '''
        
        ## Set mesh.vs to an array so that subsequent calls to asarray() on it are no-ops.
        self.vs = asarray( self.vs )
        
        self.__edges = None
        self.__halfedges = None
        self.__vertex_halfedges = None
        self.__face_halfedges = None
        self.__edge_halfedges = None
        self.__directed_edge2he_index = None
        
        self.positions_changed()
    
    def get_dangling_vertices( self ):
        '''
        Returns vertex indices in SurfaceMesh 'mesh' that belong to no faces.
        '''
        
        ## Slow:
        '''
        brute_vertex_face_valence = [ 0 ] * len( self.vs )
        for i,j,k in self.faces:
            brute_vertex_face_valence[ i ] += 1
            brute_vertex_face_valence[ j ] += 1
            brute_vertex_face_valence[ k ] += 1
        return [ i for i in xrange( len( self.vs ) ) if 0 == brute_vertex_face_valence[i] ]
        '''
        ## ~Slow
        
        ## Fast:
        '''
        brute_vertex_face_valence = zeros( len( self.vs ), dtype = int )
        self.faces = asarray( self.faces )
        brute_vertex_face_valence[ self.faces[:,0] ] += 1
        brute_vertex_face_valence[ self.faces[:,1] ] += 1
        brute_vertex_face_valence[ self.faces[:,2] ] += 1
        return where( brute_vertex_face_valence == 0 )[0]
        '''
        ## ~Fast
        
        ## Faster:
        vertex_has_face = zeros( len( self.vs ), dtype = bool )
        self.faces = asarray( self.faces )
        vertex_has_face[ self.faces.ravel() ] = True
        return where( vertex_has_face == 0 )[0]
        ## ~Faster
    
    def remove_vertex_indices( self, vertex_indices_to_remove ):
        '''
        Removes vertices in the list of indices 'vertex_indices_to_remove'.
        Also removes faces containing the vertices and dangling vertices.
        
        Returns an array mapping vertex indices before the call
        to vertex indices after the call or -1 if the vertex was removed.
        
        used
        '''
        
        ## I can't assert this here because I call this function recursively to remove dangling
        ## vertices.
        ## Also, someone manipulating the mesh might want to do the same thing (call this
        ## function on dangling vertices).
        #assert 0 == len( self.get_dangling_vertices() )
        
        
        if 0 == len( vertex_indices_to_remove ): return arange( len( self.vs ) )
        
        
        ## Slow:
        '''
        ## Make a map from old to new vertices.  This is the return value.
        old2new = [ -1 ] * len( self.vs )
        last_index = 0
        for i in xrange( len( self.vs ) ):
            if i not in vertex_indices_to_remove:
                old2new[ i ] = last_index
                last_index += 1
        
        ## Remove vertices from vs, faces, edges, and optionally uvs.
        self.vs = [ pt for i, pt in enumerate( self.vs ) if old2new[i] != -1 ]
        if hasattr( self, 'uvs' ):
            self.uvs = [ uv for i, uv in enumerate( self.uvs ) if old2new[i] != -1 ]
        ## UPDATE: We have half-edge info, so we have to call 'topology_changed()' to
        ##         regenerate the half-edge info, and 'topology_changed()' implies
        ##         'geometry_changed()', so updating anything but '.vs', '.faces'
        ##         and '.uvs' is a waste unless I can precisely update the
        ##         halfedge data structures.
        #self.__vertex_normals = asarray( [ vn for i, vn in enumerate( self.__vertex_normals ) if old2new[i] != -1 ] )
        #self.__edges = [ ( old2new[i], old2new[j] ) for i,j in self.__edges ]
        #self.__edges = [ edge for edge in self.__edges if -1 not in edge ]
        self.faces = [ ( old2new[i], old2new[j], old2new[k] ) for i,j,k in self.faces ]
        #self.__face_normals = [ n for i,n in enumerate( self.__face_normals ) if -1 not in self.faces[i] ]
        #self.__face_areas = [ n for i,n in enumerate( self.__face_areas ) if -1 not in self.faces[i] ]
        self.faces = [ tri for tri in self.faces if -1 not in tri ]
        '''
        ## ~Slow
        
        
        ## Fast:
        ## Make a map from old to new vertices.  This is the return value.
        old2new = -ones( len( self.vs ), dtype = int )
        ## Later versions of numpy.setdiff1d(), such as 2.0, return a unique, sorted array
        ## and do not assume that inputs are unique.
        ## Earlier versions, such as 1.4, require unique inputs and don't say
        ## anything about sorted output.
        ## (We don't know that 'vertex_indices_to_remove' is unique!)
        keep_vertices = sort( setdiff1d( arange( len( self.vs ) ), unique( vertex_indices_to_remove ) ) )
        old2new[ keep_vertices ] = arange( len( keep_vertices ) )
        
        ## Remove vertices from vs, faces, edges, and optionally uvs.
        ## Fast:
        self.vs = asarray( self.vs )
        self.vs = self.vs[ keep_vertices, : ]
        if hasattr( self, 'uvs' ):
            self.uvs = asarray( self.uvs )
            self.uvs = self.uvs[ keep_vertices, : ]
        
        self.faces = asarray( self.faces )
        self.faces = old2new[ self.faces ]
        self.faces = self.faces[ ( self.faces != -1 ).all( axis = 1 ) ]
        ## ~Fast
        
        
        ## Now that we have halfedge info, just call topology changed and everything but
        ## 'vs' and 'faces' will be regenerated.
        self.topology_changed()
        
        ## Remove dangling vertices created by removing faces incident to vertices in 'vertex_indices_to_remove'.
        ## We only need to call this once, because a dangling vertex has no faces, so its removal
        ## won't remove any faces, so no new dangling vertices can be created.
        dangling = self.get_dangling_vertices()
        if len( dangling ) > 0:
            old2new_recurse = self.remove_vertex_indices( dangling )
            assert 0 == len( self.get_dangling_vertices() )
            
            '''
            for i in xrange( len( old2new ) ):
                if -1 != old2new[i]: old2new[i] = old2new_recurse[ old2new[ i ] ]
            '''
            old2new[ old2new != -1 ] = old2new_recurse[ old2new ]
        
        return old2new
    
    def remove_face_indices( self, face_indices_to_remove ):
        '''
        Removes faces in the list of indices 'face_indices_to_remove'.
        Also removes dangling vertices.
        
        Returns an array mapping face indices before the call
        to face indices after the call or -1 if the face was removed.
        
        used
        '''
        
        if 0 == len( face_indices_to_remove ): return arange( len( self.faces ) )
        
        
        ## Fast:
        ## Make a map from old to new faces.  This is the return value.
        old2new = -ones( len( self.faces ), dtype = int )
        ## Later versions of numpy.setdiff1d(), such as 2.0, return a unique, sorted array
        ## and do not assume that inputs are unique.
        ## Earlier versions, such as 1.4, require unique inputs and don't say
        ## anything about sorted output.
        ## (We don't know that 'face_indices_to_remove' is unique!)
        keep_faces = sort( setdiff1d( arange( len( self.faces ) ), unique( face_indices_to_remove ) ) )
        old2new[ keep_faces ] = arange( len( keep_faces ) )
        
        ## Remove vertices from vs, faces, edges, and optionally uvs.
        ## Fast:
        self.faces = asarray( self.faces )
        self.faces = self.faces[ keep_faces, : ]
        ## ~Fast
        
        
        ## Now that we have halfedge info, just call topology changed and everything but
        ## 'vs' and 'faces' will be regenerated.
        self.topology_changed()
        
        ## Remove dangling vertices created by removing faces incident to vertices.
        ## Since we are only removing dangling vertices, 'self.faces' can't be affected,
        ## so we don't need to worry about the 'old2new' map.
        dangling = self.get_dangling_vertices()
        if len( dangling ) > 0:
            self.remove_vertex_indices( dangling )
            assert 0 == len( self.get_dangling_vertices() )
        
        return old2new
    
    
    def append( self, mesh ):
        '''
        Given a mesh, with two properties,
            .vs, containing a list of 3d vertices
            .faces, containing a list of triangles as triplets of indices into .vs
        appends 'mesh's vertices and faces to self.vs and self.faces.
        '''
        
        ## mesh's vertices are going to be copied to the end of self.vs;
        ## All vertex indices in mesh.faces will need to be offset by the current
        ## number of vertices in self.vs.
        vertex_offset = len( self.vs )
        
        self.vs = list( self.vs ) + list( mesh.vs )
        self.faces = list( self.faces ) + list( asarray( mesh.faces, dtype = int ) + vertex_offset )
        
        
        ## If there are uvs, concatenate them.
        
        ## First, if self is an empty mesh (without uv's), and the mesh to append-to has uv's,
        ## create an empty .uvs property in self.
        if not hasattr( self, 'uvs' ) and hasattr( mesh, 'uvs' ) and len( self.vs ) == 0:
            self.uvs = []
        
        if hasattr( self, 'uvs' ) and hasattr( mesh, 'uvs' ):
            self.uvs = list( self.uvs ) + list( mesh.uvs )
        elif hasattr( self, 'uvs' ):
            del self.uvs
        
        
        ## We're almost done, we only need to call topology_changed().
        ## However, let's see if we can keep some properties that are slow to regenerate.
        self__face_normals = self.__face_normals
        self__face_areas = self.__face_areas
        self__vertex_normals = self.__vertex_normals
        self__vertex_areas = self.__vertex_areas
        
        self.topology_changed()
        
        if self__face_normals is not None and mesh.__face_normals is not None:
            self.__face_normals = append( self__face_normals, mesh.__face_normals, axis = 0 )
        if self__face_areas is not None and mesh.__face_areas is not None:
            self.__face_areas = append( self__face_areas, mesh.__face_areas, axis = 0 )
        if self__vertex_normals is not None and mesh.__vertex_normals is not None:
            self.__vertex_normals = append( self__vertex_normals, mesh.__vertex_normals, axis = 0 )
        if self__vertex_areas is not None and mesh.__vertex_areas is not None:
            self.__vertex_areas = append( self__vertex_areas, mesh.__vertex_areas, axis = 0 )

    def GetBoundaryComponents(self):
        visited_hes = set()
        bdry_comp_hes = []
        if self.__halfedges is None: self.update_halfedges()

        for he in self.__halfedges:
            he_idx = self.OppositeHE(he).opposite_he
            if he_idx in visited_hes:
                continue
            else:
                visited_hes.add(he_idx)
                
            if he.face == -1:
                new_bdry = [he]
                init_he = he
                init_idx = self.OppositeHE(init_he).opposite_he
                current_he = self.NextHE(he)
                while current_he != init_he:
                    current_idx = self.OppositeHE(current_he).opposite_he
                    new_bdry.append(current_he)
                    visited_hes.add(current_idx)
                    current_he = self.NextHE(current_he)
                bdry_comp_hes.append(new_bdry)
        return bdry_comp_hes
    
    def GetBoundaryComponentVertices(self):
        bdry_comp_hes = self.GetBoundaryComponents()
        bdry_comp_verts = []
        for bdry_hes in bdry_comp_hes:
            bdry_verts = [self.FromVertex(bdry_hes[0])]
            for he in bdry_hes:
                bdry_verts.append(self.ToVertex(he))
            bdry_comp_verts.append(bdry_verts)
        return bdry_comp_verts
    
    def FromSurfaceMeshes( meshes ):
        '''
        Given a sequence of meshes, each with two properties,
            .vs, containing a list of 3d vertices
            .faces, containing a list of quads as quads of indices into .vs
        returns a single SurfaceMesh object containing all meshes concatenated together.
        '''
        
        result = SurfaceMesh()
        for mesh in meshes:
            result.append( mesh )
        
        ## Reset the lifetime counter
        result.lifetime_counter = 0
        return result
    
    FromSurfaceMeshes = staticmethod( FromSurfaceMeshes )
    
    
    def FromOBJ_FileName( obj_fname ):
        if obj_fname.endswith( '.gz' ):
            import gzip
            f = gzip.open( obj_fname )
        else:
            f = open( obj_fname )
        return SurfaceMesh.FromOBJ_Lines( f )
    
    FromOBJ_FileName = staticmethod( FromOBJ_FileName )
    
    
    def FromOBJ_Lines( obj_lines ):
        '''
        Given lines from an OBJ file, return a new SurfaceMesh object.
        
        tested
        '''
        
        result = SurfaceMesh()
        no_uvs = True
        only_triangles = True
        
        ## NOTE: We only handle faces and vertex positions.
        for line in obj_lines:
            line = line.strip()
            
            sline = line.split()
            ## Skip blank lines
            if not sline: continue
            
            elif sline[0] == 'v':
                result.vs.append( [ float(v) for v in sline[1:] ] )
                ## Vertices must have three coordinates.
                ## UPDATE: Let's be flexible about this.
                # assert len( result.vs[-1] ) == 3
            
            elif sline[0] == 'vt':
                if no_uvs:
                    result.uvs = []
                    no_uvs = False
                result.uvs.append( [ float(v) for v in sline[1:] ] )
            
            elif sline[0] == 'f':
                ## The split('/')[0] means we record only the vertex coordinate indices
                ## for each face.
                face_vertex_ids = [ int( c.split('/')[0] ) for c in sline[1:] ]
                ## Faces must be quadrilaterals.
                assert len( face_vertex_ids ) in [3,4]
                if len(face_vertex_ids)==3:
                    face_vertex_ids.append(inf)
                
                ## Face vertex indices cannot be zero.
                ## UPDATE: Do this assert once at the end. The following code
                ##         will convert the 0 to -1.
                # assert not any([ ind == 0 for ind in face_vertex_ids ])
                
                ## Subtract one from positive indices, and use relative addressing for negative
                ## indices.
                face_vertex_ids_mod = []
                for ind in face_vertex_ids:
                    if isinf(ind):
                        face_vertex_ids_mod.append(-1)
                        result.is_all_quad_mesh = False
                    elif ind >= 0:
                        face_vertex_ids_mod.append(ind-1)
                    else:
                        face_vertex_ids_mod.append(len(result.vs) + ind)
                # face_vertex_ids = [
                #     ( ind-1 ) if ( ind >= 0 ) else ( len(result.vs) + ind )
                #     for ind in face_vertex_ids
                #     ]
                
                ## UPDATE: Do this assert once at the end.
                # assert all([ ind < len( result.vs ) for ind in face_vertex_ids ])
                result.faces.append( face_vertex_ids_mod )
                
                if result.faces[-1][3] >= 0:
                    only_triangles = False
        
        result.is_all_tri_mesh = only_triangles
        result.vs = asarray( result.vs )
        if hasattr(result, 'uvs'):
            result.uvs = asarray( result.uvs )
        result.faces = asarray( result.faces, dtype = int )
        assert (result.faces < len( result.vs ) ).all()
        
        return result
    
    FromOBJ_Lines = staticmethod( FromOBJ_Lines )
    
    
    def AppendSurfaceBoundarySets(self, fname):
        
        if fname == '':
            bdry_verts = self.GetBoundaryComponentVertices()
            self.boundary_set_vert_idxs = [set(bdry_vs) for bdry_vs in bdry_verts]
            self.first_boundary_vert_idxs = [(bdry_vs[0],bdry_vs[1]) for bdry_vs in bdry_verts]
            
        else:
                
        
            with open(fname, 'r') as file:
                
                verts = []
                bdry_set_idxs = []
                for line in file:
                    if line[0] == 'v':
                        splitline = line.strip().split(" ")
                        verts.append([float(splitline[i]) for i in range(1,4)])
                    
                    if line[0] == 'l':
                        splitline = line.strip().split(" ")
                        # -1 to index because OBJ file formatting vs Python
                        bdry_set_idxs.append([int(idx)-1 for idx in splitline[1:]])
        
            mesh_bdry_idxs = []
            first_bdry_idxs = []
            all_bdry_verts = self.boundary_vertices()
            for line_vert_set in bdry_set_idxs:
                init_v_idx = -1
                line_vert_idx = line_vert_set[0]
                init_vert = verts[line_vert_idx]
                for v_idx in all_bdry_verts:
                    v = self.vs[v_idx]
                    if mag2(v -init_vert) < 1e-8:
                        init_v_idx = v_idx
                        break
                if init_v_idx == -1:
                    sys.exit("Trying to add a boundary vertex that is not on this mesh boundary")
                
                
                mesh_bdry = []
                mesh_bdry.append(init_v_idx)
                for i in range(1,len(line_vert_set)):
                    vertex_is_found = False
                    # find the vertex that matches the next line vertex
                    next_vert_idx = line_vert_set[i]
                    this_line_vertex = verts[next_vert_idx]
                    adj_verts = self.vertex_vertex_neighbors(init_v_idx)
                    for j in adj_verts:
                        vert_pos = self.vs[j]
                        if mag2(vert_pos - this_line_vertex) < 1e-8:
                            mesh_bdry.append(j)
                            init_v_idx = j
                            vertex_is_found = True
                            break
                    if not vertex_is_found:
                        sys.exit("No adjacency found for the specified vertex")
                    
                first_bdry_idxs.append((mesh_bdry[0],mesh_bdry[1]))
                mesh_bdry_idxs.append(set(mesh_bdry))
                
            self.boundary_set_vert_idxs = mesh_bdry_idxs
            self.first_boundary_vert_idxs = first_bdry_idxs
            
        return self.boundary_set_vert_idxs
                    
    
    def write_OBJ( self, fname, header_comment = None ):
        if hasattr( self, 'uvs' ):
            write_OBJ(self.vs,self.faces,fname,self.uvs,header_comment)
        else:
            write_OBJ(self.vs,self.faces,fname,None,header_comment)
        
    def write_OFF( self, fname ):
        '''
        Writes the data out to an OFF file named 'fname'.
        '''
        
        out = open( fname, 'w' )
        
        out.write( 'OFF\n' )
        out.write( '%d %d 0\n' % ( len( self.vs ), len( self.faces ) ) )
        
        for v in self.vs:
            out.write( '%r %r %r\n' % tuple(v) )
        for f in self.faces:
            if(f[3]<0):
                out.write( '3 %s %s %s\n' % ( f[0], f[1], f[2] ) )
            else:
                out.write( '4 %s %s %s %s\n' % tuple(f) )
        
        out.close()
        
        print( 'OFF written to:', fname)

    def GetFaceCentroid(self, face_idx):
        if self.__face_centroid == None:
            self.__face_centroid = []
            for i in range(0,len(self.faces)):
                vert_idxs = self.faces[i]
                number_of_vertices = 0
                centroid = zeros(len(self.vs[0]))
                for j in vert_idxs:
                    if j == -1:
                        break
                    centroid += self.vs[j]
                    number_of_vertices += 1
                self.__face_centroid.append(centroid / number_of_vertices)
            
        return self.__face_centroid[face_idx]
    
    def GetFaceArea(self, face_idx):
        # TODO: this is incorrect if quadriaterals have vertices that are non-planar
        # see https://math.stackexchange.com/questions/3049969/calculate-surface-normal-and-area-for-a-non-planar-quadrilateral
        f_hes = self.get_face_halfedges(face_idx)
        he0 = f_hes[0]
        he1 = f_hes[1]
        
        he0_vec = self.HalfEdgeToVector(he0)
        he1_vec = self.HalfEdgeToVector(he1)
        
        area_vec = cross(he0_vec,he1_vec)
        len_area_vec = linalg.norm(area_vec)
        if self.is_triangle_face(face_idx):
            return 0.5 * len_area_vec
        else:
            he2 = f_hes[2]
            he3 = f_hes[3]
            
            he2_vec = self.HalfEdgeToVector(he2)
            he3_vec = self.HalfEdgeToVector(he3)
            
            area_vec2 = cross(he2_vec,he3_vec)
            len_area_vec2 = linalg.norm(area_vec2)
            
            return 0.5 * (len_area_vec + len_area_vec2)
            
        

    def ReindexMesh(self, init_vert_idx, save_name = ""):
        
        v_map = {}
        f_map = {}
        reverse_v_map = []
        reverse_f_map = []
        
        # create a vertex queue with the initial vertex index in it
        v_queue = [init_vert_idx]
        
        # do a breadth-first search from the initial vertex index
        counter = 0
        while len(v_queue) > 0:
            v = v_queue.pop()
            if v in v_map: # ignore vertices that have already been accounted for
                continue
            
            v_map[v] = counter
            reverse_v_map.append(v)
            counter += 1
            v_queue.extend(self.vertex_vertex_neighbors(v))
        
        f_queue = self.vertex_face_neighbors(init_vert_idx)
        counter = 0
        while len(f_queue) > 0:
            f = f_queue.pop()
            if f in f_map:
                continue
            
            f_map[f] = counter
            reverse_f_map.append(f)
            counter += 1
            f_queue.extend(self.face_face_neighbors(f))
            
        new_vs = zeros(self.vs.shape)
        new_faces = zeros(self.faces.shape,dtype='int')
        
        for i in range(0,new_vs.shape[0]):
            new_vs[i] = self.vs[reverse_v_map[i]]
        for i in range(0,new_faces.shape[0]):
            idxs = self.faces[reverse_f_map[i]]
            for j in range(0,4):
                if idxs[j] == -1:
                    new_faces[i,j] = -1
                else:
                    new_faces[i,j] = v_map[idxs[j]]
            
        if save_name != "":
            write_OBJ(new_vs,new_faces,save_name)
            
        return new_vs,new_faces
    
    def FindClosestVertex(self, coordinates):
        min_dist = infty
        min_idx = -1
        for j in range(0,len(self.vs)):
            v = self.vs[j]
            print(j,v)
            vector = [v[i] - coordinates[i] for i in range(0,3)]
            temp_dist = mag2(vector)
            if temp_dist < min_dist:
                min_dist = temp_dist
                min_idx = j
                
        return min_idx

def write_OBJ(vs,faces,fname, uvs = None, header_comment = None ):
    '''
    Writes the data out to an OBJ file named 'fname'.
    Optional comment 'header_comment' is printed at the
    top of the OBJ file, after prepending the OBJ comment
    marker at the head of each line.
    
    tested
    '''
    
    
    ## Estimate for mesh size:
    ## 16 bytes for a vertex row,
    ## optionally 16 bytes for a uv row,
    ## 12/20 bytes for a face row with/without uv's.
    ## Assuming no uv's and 2 faces per vertex,
    ## a 1MB mesh is made of (1024*1024/(16+2*12)) = 26214 vertices.
    ## If we have uv's, then we will reach 1MB with (1024*1024/(2*16+2*20)) = 14563 vertices.
    ## Print a warning if we're going to save a mesh much larger than a megabyte.
    if len( vs ) > 15000:
        print( 'Writing a large OBJ to:', fname )
    
    
    out = open( fname, 'w' )
    
    if header_comment is None:
        import sys
        header_comment = 'Written by ' + ' '.join([ arg.replace('\n',r'\n') for arg in sys.argv ])
    
    ## Print the header comment.
    for line in header_comment.split('\n'):
        out.write( '## %s\n' % (line,) )
    out.write( '\n' )
    
    
    ## Print vertices.
    for v in vs:
        out.write( 'v %r %r %r\n' % tuple(v) )
    out.write( '\n' )
    
    
    ## Print uv's if we have them.
    if uvs != None:
        for uv in uvs:
            out.write( 'vt %r %r\n' % tuple(uv) )
        out.write( '\n' )
        
        ## Print faces with uv's.
        for f in faces:
            if (f[3]<0):
                out.write( 'f %s/%s %s/%s %s/%s\n' % ( f[0]+1,f[0]+1, f[1]+1,f[1]+1, f[2]+1,f[2]+1 ) )
                #out.write( 'f %s/%s %s/%s %s/%s\n' % tuple( ( asarray(f,dtype=int) + 1 ).repeat(2) ) )
            else:
                out.write( 'f %s/%s %s/%s %s/%s %s/%s\n' % ( f[0]+1,f[0]+1, f[1]+1,f[1]+1, f[2]+1,f[2]+1, f[3]+1,f[3]+1 ) )
    else:
        ## Print faces without uv's.
        for f in faces:
            #out.write( 'f %s %s %s\n' % tuple(asarray(f,dtype=int) + 1) )
            if f[3]<0:
                out.write( 'f %s %s %s\n' % ( f[0]+1, f[1]+1, f[2]+1 ) )
            else:
                out.write( 'f %s %s %s %s\n' % ( f[0]+1, f[1]+1, f[2]+1, f[3]+1 ) )
    
    
    out.close()
    
    print( 'OBJ written to:', fname)


def PlotSurfaceMesh(mesh,face_values=[],plot3d=True,showplot = True):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np


    if plot3d:
        ax = plt.figure().add_subplot(projection='3d')
    else:
        ax = plt.figure().add_subplot()
# ax = fig.gca(projection='3d')

    Q = mesh.faces
    T = []
    split_face_values = []
    for i in range(0,len(Q)):
        face = Q[i]
        if face[-1]!=-1:
            # T.append([face[0],face[1],face[3]])
            # T.append([face[0],face[2],face[3]])
            T.append([face[0],face[1],face[2]])
            T.append([face[0],face[2],face[3]])
            if len(face_values) > 0:
                split_face_values.append(face_values[i])
                split_face_values.append(face_values[i])
        else:
            T.append(face[0:3])
            if len(face_values) > 0:
                split_face_values.append(face_values[i])
    vertices = mesh.vs
    triang = mtri.Triangulation(vertices[:,0], vertices[:,1], triangles=T)
    if plot3d:
        p3dc = ax.plot_trisurf(triang, vertices[:,2], edgecolor=[[0,0,0]], linewidth=1.0, shade=True, antialiased=True)
        
    else:
        if split_face_values == []:
            split_face_values = [0 for f in triang.triangles]
        p3dc = ax.tripcolor(triang,split_face_values, linewidth=1.0)

    if face_values != []:
        cmap='viridis'
        from matplotlib.cm import ScalarMappable, get_cmap
        from matplotlib.colors import Normalize

        norm = Normalize()
        colors = get_cmap(cmap)(norm(split_face_values))

        # set the face colors of the Poly3DCollection
        p3dc.set_fc(colors)

        mappable = ScalarMappable(cmap=cmap, norm=norm)
        # mappable.set_array(face_values)
        # ########## change the face colors ####################
        mappable = map_colors(p3dc, lambda x,y,z:x+y+z, 'Spectral')
        # # ####################################################
    
        # possibly add a colormap
        plt.colorbar(mappable, shrink=0.67, aspect=16.7)

    # we are done
    if showplot:
        plt.show()
        
    return ax


def PlotSurfaceMeshWithBoundaryComps(mesh,face_values=[],plot3d=True):
    
    scattertype = ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    bdries = mesh.boundary_set_vert_idxs
    
    ax = PlotSurfaceMesh(mesh,face_values,plot3d,showplot=False)
    
    counter = 0
    for bdry in bdries:
        m = scattertype[counter]
        
        xs = []
        ys = []
        zs = []
        for idx in bdry:
            v = mesh.vs[idx]
            xs.append(v[0])
            ys.append(v[1])
            zs.append(v[2])
        
        if plot3d:
            ax.scatter(xs, ys, zs, marker=m,label="bdry"+str(counter))     
        else:
            ax.scatter(xs, ys, marker=m,label="bdry"+str(counter))
        counter += 1
    
    ax.legend(loc=1)

def map_colors(p3dc, func, cmap='viridis'):
    """
Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """
    
    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize
    from numpy import array

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = array([array((x[s],y[s],z[s])).T for s in slices])

    # compute the barycentres for each triangle
    xb, yb, zb = triangles.mean(axis=1).T
    
    # compute the function in the barycentres
    values = func(xb, yb, zb)

    # usual stuff
    norm = Normalize()
    colors = get_cmap(cmap)(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)


def PlotTriangulationSolution(mesh,DTotal):
    X = mesh.vs[:,0]
    Y = mesh.vs[:,1]
    faces = []
    for i in range(0,len(mesh.faces)):
        if mesh.is_quadrilateral_face(i):
            faces.append([mesh.faces[i][0],\
                          mesh.faces[i][1],\
                          mesh.faces[i][2]])
            faces.append([mesh.faces[i][2],\
                          mesh.faces[i][3],\
                          mesh.faces[i][0]])
        else:
            faces.append([mesh.faces[i][0],\
                          mesh.faces[i][1],\
                          mesh.faces[i][2]])
    
    
    array_faces = array(faces,dtype=int)
    
    triangles = mtri.Triangulation(X, Y, triangles=array_faces)
    
    matplotlib.rcParams['figure.dpi'] = 300
    
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tcf = ax1.tricontourf(triangles, DTotal)
    fig1.colorbar(tcf)
    ax1.tricontour(triangles, DTotal, colors='k')
    ax1.set_title('Contour plot of distribution')
    
    ax1.triplot(triangles, color='0.7',linewidth=0.1)

    

    
def PlotSurfaceMeshIndexing(mesh):
    for j in range(0,len(mesh.faces)):
        face = mesh.faces[j]
        xcoords = []
        ycoords = []
        for i in range(0,len(face)):
           if mesh.faces[j][i] == -1: #Make final (fourth) x- and y-coordinates the same as first x- and y-coordinates if face is a triangle
               pass
           else:
               xcoords.append(mesh.vs[face[i]][0])
               ycoords.append(mesh.vs[face[i]][1])
        mx = mean(xcoords)
        my = mean(ycoords)
        label = "f{:d}".format(j)
        xcoords.append(mesh.vs[face[0]][0])
        ycoords.append(mesh.vs[face[0]][1])
        offset = (0,0)
        plt.plot(xcoords,ycoords,color="black")
        plt.annotate(label, # this is the text
              (mx,my), # these are the coordinates to position the label
              textcoords="offset points", # how to position the text
              xytext=offset, # distance from text to points (x,y)
              ha='center', # horizontal alignment can be left, right or center
              size=20)
        
        n_verts = 3 if face[-1]==-1 else 4
        for i in range(0,n_verts):
            x = 0
            y = 0
            for k in range(0,n_verts):
                if k == i:
                    x += xcoords[k]*2
                    y += ycoords[k]*2
                else:
                    x += xcoords[k]/3
                    y += ycoords[k]/3
            x = x / 3
            y = y / 3
            label = "{:d}".format(i)
            offset = (0,0)
            plt.annotate(label, # this is the text
                  (x,y), # these are the coordinates to position the label
                  textcoords="offset points", # how to position the text
                  xytext=offset, # distance from text to points (x,y)
                  ha='center', # horizontal alignment can be left, right or center
                  color='blue',
                  size=20)

    
    for i in range(0,len(mesh.vs)):
        vert = mesh.vs[i]
        x = vert[0]
        y = vert[1]
        plt.scatter([x],[y],marker='o', s=30, color="red")
        label = "v{:d}".format(i)
        offset = (0,0)
        plt.annotate(label, # this is the text
              (x,y), # these are the coordinates to position the label
              textcoords="offset points", # how to position the text
              xytext=offset, # distance from text to points (x,y)
              ha='center', # horizontal alignment can be left, right or center
              color='darkred',
              size=20)

## We can't pickle anything that doesn't have a name visible at module scope.
## In order to allow pickling of class SurfaceMesh, we'll make a reference to the inner HalfEdge class
## here at the module level.
HalfEdge = SurfaceMesh.HalfEdge

def main():
    import sys
    if len( sys.argv ) > 1:
        mesh = SurfaceMesh.FromOBJ_FileName( sys.argv[1] )
    # mesh.write_OBJ( sys.argv[2] )

if __name__ == '__main__':
    main()

def PlotBasic():    
    mesh=SurfaceMesh.FromOBJ_FileName("mixed_mesh.obj")
    PlotSurfaceMeshIndexing(mesh)
    # PlotSurfaceMesh(mesh)
    
    # print(mesh.FindClosestVertex([3,4,7]))
    
def PlotAdvanced():    
    mesh=SurfaceMesh.FromOBJ_FileName("wide_angle_flange.obj")
    face_areas = []
    for i in range(0,len(mesh.faces)):
        face_areas.append(mesh.GetFaceArea(i))
    
    PlotSurfaceMesh(mesh,face_areas)

def PlotBrick():    
    mesh=SurfaceMesh.FromOBJ_FileName("brick_obj.obj")
    my_bdries = mesh.AppendSurfaceBoundarySets("brick_boundaries.obj")
    face_areas = []
    # for i in range(0,len(mesh.faces)):
    #     face_areas.append(mesh.GetFaceArea(i))
    
    plot_3d = False
    # PlotSurfaceMesh(mesh,face_areas,plot_3d)
    
    print((mesh.vs))
    # PlotSurfaceMeshIndexing(mesh)
    PlotSurfaceMeshWithBoundaryComps(mesh,face_areas,plot_3d)



def PlotBlock():    
    mesh=SurfaceMesh.FromOBJ_FileName("unit_square.obj")
    my_bdries = mesh.AppendSurfaceBoundarySets("unit_square_boundaries.obj")
    face_areas = []
    # for i in range(0,len(mesh.faces)):
    #     face_areas.append(mesh.GetFaceArea(i))
    
    plot_3d = False
    # PlotSurfaceMesh(mesh,face_areas,plot_3d)
    PlotSurfaceMeshIndexing(mesh)
    PlotSurfaceMeshWithBoundaryComps(mesh,face_areas,plot_3d)


# PlotBasic()
# plt.show()


# PlotBrick()
# plt.show()

# PlotBlock()
# plt.show()