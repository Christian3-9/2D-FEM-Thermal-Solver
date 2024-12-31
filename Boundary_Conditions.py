
"""
@author: Christian Valencia Narva

"""

import sys
from enum import Enum
from abc import ABCMeta, abstractmethod 

class BoundaryConditionType(Enum):
    DIRICHLET = 0
    NEUMANN = 1
    ROBIN = 2

class BoundaryCondition(object, metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self,boundarycondition_type,boundary_nodes):
        self.type = boundarycondition_type
        self.bdry_nodes = boundary_nodes
        
class DirichletBoundaryCondition(BoundaryCondition):
    
    def __init__(self,boundarycondition_type,boundary_nodes,uval = lambda x,y:0):
        super().__init__(boundarycondition_type, boundary_nodes)
        self.rhs_func = uval
        self.bdry_coeffs = {} 
        self.element_IEN=set()
        
# complete this class
class RobinBoundaryCondition(BoundaryCondition):
    def __init__(self, boundarycondition_type, boundary_nodes,u_multiplier = lambda x,y: 0, uval= lambda x,y: 0):
        super().__init__(boundarycondition_type, boundary_nodes)
        self.u_multiplier= u_multiplier
        self.rhs_func= uval
        self.element_IEN=set()

# complete this class      
class NeumannBoundaryCondition(BoundaryCondition):
    def __init__(self, boundarycondition_type, boundary_nodes, flux = lambda x,y: 0):
        super().__init__(boundarycondition_type, boundary_nodes)
        self.rhs_func=flux
        self.element_IEN=set()
        #self.flux=flux



