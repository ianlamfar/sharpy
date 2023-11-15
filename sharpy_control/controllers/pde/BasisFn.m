% Function to compute the basis functions for Galerkin's method
% Arguments: DoF (1 for twist); index > 0; derivative; coordinate
function fn = BasisFn(arg1,k,d,z) 

if k == 0
    if d == 0
        eta = 0.78; epsil = 0.02;
    fn = (0.5/(1-eta + epsil))*(1 + tanh(10*(z-eta))) ...
        +  (0.5/(1-eta + epsil))*(1 - tanh(10*(z-1+epsil)));
    else fn = 1;
    end
else
	
    if arg1 == 1
		lk = (2*k-1)*pi/2;
        repo_t = [sin(lk*z),cos(lk*z),-sin(lk*z)];
        fn = sqrt(2)*(lk^d)*repo_t(d+1);
    else % if arg1 = 2 for bending
		if k == 1
			lk = 1.8745;
		elseif k == 2
			lk = 4.6941;
		else
			lk = (2*k-1)*pi/2;
		end
		Bterm = (sinh(lk) - sin(lk))/(cos(lk) + cosh(lk));
        repo_b = [(cos(lk*z) - cosh(lk*z) - Bterm*(sin(lk*z)-sinh(lk*z))),...
    	(-sin(lk*z) - sinh(lk*z) - Bterm*(cos(lk*z) - cosh(lk*z))),...
    	(-cos(lk*z) - cosh(lk*z) + Bterm*(sin(lk*z) + sinh(lk*z))),...
    	(sin(lk*z) - sinh(lk*z) + Bterm*(cos(lk*z) + cosh(lk*z))),...
    	(cos(lk*z) - cosh(lk*z) - Bterm*(sin(lk*z)-sinh(lk*z)))];
        fn = (lk^d)*repo_b(d+1);
    end
   % disp([arg1, k, d, z])
end
