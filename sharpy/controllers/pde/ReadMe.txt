
Main file: Script2Dof.m
Prepares the mass and stiffness matrix, as well as other force matrix, using
Galerkin's method. Sets up the problem.

Function files:
(1) FT2Dof.m: contains the ODEs for wing vibration. Uses the matrices put together by Script2Dof

(2) matrix_element.m: executes Galerkin's method for determining the each element of a matrix. 

(3) BasisFn.m: computes the basis functions needed for Galerkin's method. Exact mode shapes of the 
unforced structural ODE as used as basis functions.


Redundant Files (Do Not Call or Use in your scripts):
Events.m
TestingFunctions.m
