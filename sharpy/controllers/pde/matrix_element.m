% Function to create the mass and stiffness matrices
% Function computes int_0^1 phi_1 phi_2 dz
function aij = matrix_element(v1,v2)
no_of_ele = 100;
dz = 0.01;
aij = 0.0;
for i = 1:no_of_ele
    z = (i-1)*dz + dz/2;
    t1 = BasisFn(v1(1),v1(2),v1(3),z);
    % disp([t1, z])
    t2 = BasisFn(v2(1),v2(2),v2(3),z);
    aij = aij + t1*t2*dz;
    % disp(aij)
end
