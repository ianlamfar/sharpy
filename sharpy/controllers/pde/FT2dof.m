function dx = FT2dof(t,x)
global M K Daint Lint Fmat Fmat2
global n_t n_b par
%global Ip GJ
n = n_t + n_b;
c = par(1);
b = par(2);
xa = par(3);
V = par(4);
Cla = 2*pi/(1 + c/b);
Minf = V/340;
f = 0.61*V*V*c*Cla/sqrt(1 - Minf^2);  % capital M in the paper

eta = 4*(10^(-4)); % Kelvin-Voigt damping factor

%da = 0.0; % a static deflection for now.
ref_val = 0.1;

%With aileron feedback
da = I_control + P_control
da = -0.02*x(end) - 0.1*(Lint'*x(1:n_t) - ref_val);  n_t = number of torsion mode, write ref_val as alpha (rad) * span
da = sign(da)*min(0.2, abs(da));

%open loop
%da = 0;

fda = 0.61*V*V*c*1.7*da*Daint/sqrt(1 - Minf^2);

dx = zeros(2*n+1,1);

xidot = x(n+n_t+1:2*n);
Q1 = M\[xa*c*eye(n_t),zeros(n_t,n_b); zeros(n_b,n_t),-eye(n_b)];

dx(1:n) = x(n+1:2*n,1);
% M\K is what comes out of a
dx(n+1:2*n) = -(M\K)*x(1:n) - eta*(M\K)*x(n+1:2*n) ...
    + f*Q1*(Fmat*x(1:n_t) + (1/V)*Fmat2*xidot) ...
    + fda;
dx(end) = Lint'*x(1:n_t) - ref_val;  
