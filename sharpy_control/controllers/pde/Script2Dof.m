% Program to simulate a 2-dof flexible wing
clear
close all

global M K Daint Lint Fmat Fmat2 % Daint: effect of an aileron deflection; Lint: integral of theta
global n_b n_t par
global Ip GJ

% Define all wing parameters
L = 30;
Vinf = 300;
chord = 5.8;
mass_wing = 50000;
thickness = 0.07*chord;
E = 80; %in GPA
xe = 0.05; % distance between CG and shear centre
xa = 0.4; % between AC and shear centre, fraction of chord

Ip = (1/16)*(mass_wing/L)*chord^2;
Iyy = (pi/64)*chord*thickness^3;
J = pi*chord*thickness^3/16;
G = E/2.5;
GJ = (10^9)*G*J/L^2;
EI = (10^9)*E*Iyy/L^4;

clear E G J Iyy thickness
n_t = 3; n_b = 3;

%% Create mass matrix

% Notation for matrix_element: [a, b, c]
% a = 1 for twist; 2 for bending; any number for aileron deflection
% b = mode number (1, 2, 3, ...); 0 for aileron deflection
% c = derivative (0, 1, 2, ...); 0 for aileron deflection
% To get a constant: use a = 1; b = 0; c = 1.

M = zeros(n_t + n_b);
disp(size(M))
for i = 1:n_t
    for j = 1:n_t
    M(i,j) = Ip*matrix_element([1,i,0],[1,j,0]);
    end
end
for i = 1:n_b
    for j = 1:n_b
    M(n_t+i,n_t+j) = (mass_wing/L)*matrix_element([2,i,0],[2,j,0]);
    end
end
for i = 1:n_t
    for j = 1:n_b
        M(i,n_t+j) = -(mass_wing/L)*xe*chord*matrix_element([1,i,0],[2,j,0]);
        M(n_t+j,i) = M(i,n_t+j);
    end
end

%% Create the stiffness matrix (K) which can be used with K-V damping
K = zeros(n_t + n_b);
for i = 1:n_t
    for j = 1:n_t
    K(i,j) = -GJ*matrix_element([1,j,2],[1,i,0]);
    end
end
for i = 1:n_b
    for j = 1:n_b
    K(n_t+i,n_t+j) = EI*matrix_element([2,j,4],[2,i,0]);
    end
end

% Matrix for adding the effect of aileron deflection; assumes a 20% long aileron
Daint = zeros(n_t+n_b,1); 
for i = 1:n_t
    Daint(i) = xa*chord*matrix_element([1,0,0],[1,i,0]);
end
for i = 1:n_b
    Daint(n_t + i) = -matrix_element([1,0,0],[2,i,0]);
end

% Matrix for evaluating the modal forcing function
Fmat = zeros(n_t+n_b, n_t);
for i = 1:n_t
    for j = 1:n_t
    Fmat(i,j) = matrix_element([1,i,0],[1,j,0]);
    end
end
for i = 1:n_b
    for j = 1:n_t
    Fmat(n_t+i,j) = matrix_element([2,i,0],[1,j,0]);
    end
end

Fmat2 = zeros(n_t+n_b, n_t);
for i = 1:n_t
    for j = 1:n_b
    Fmat2(i,j) = matrix_element([1,i,0],[2,j,0]);
    end
end
for i = 1:n_b
    for j = 1:n_b
    Fmat2(n_t+i,j) = matrix_element([2,i,0],[2,j,0]);
    end
end


% Matrix for calculating integral of theta for lift estimation
Lint = zeros(n_t, 1);
for i = 1:n_t
    Lint(i) = matrix_element([1,i,0],[1,0,1]);
end

% Matrix for computing the bending displacements

% Test the time constants
n = n_t + n_b;
Cla = 2*pi/(1 + chord/L);
Minf = Vinf/340;
f = 0.61*Vinf*Vinf*chord*Cla/sqrt(1 - Minf^2);
Q1 = f*(M\[xa*chord*eye(n_t),zeros(n_t,n_b); zeros(n_b,n_t),-eye(n_b)]);
test_mat = [zeros(n), eye(n); (-(M\K) + Q1*[Fmat, zeros(n,n_b)]),(0*-4*10^(-4)*(M\K) + Q1*[zeros(n,n_t),Fmat2])];
test_mat_short = [zeros(n_t), eye(n_t); -(M(1:n_t,1:n_t)\K(1:n_t,1:n_t)), -2*10^(-5)*(M(1:n_t,1:n_t)\K(1:n_t,1:n_t))];
disp('Time constants:')
disp(eig(test_mat))
disp('Of which the twisting dynamics:')
disp(eigs(test_mat_short))
% Set up the integration
par = [chord,L,xa,Vinf]; 
x0 = zeros(2*n+1,1); % end for lift error
for c1 = 1:n_t
    x0(c1) = 0.04/(c1)^2*rand(1);
end
for c2 = 1:n_b
    x0(n_t+c2) = 0.04/(c2)^2*rand(1);
end

%% Simulation
%options = odeset('Events',@Events); % stop if the solutions diverge
%xsol = lsode(@FT2dof,[0,10],x0);
%options = odeset('Events',@large_def);

%[tsol,xsol] = ode23s(@FT2dof,[0,10],x0,options);
[tsol,xsol] = ode23s(@FT2dof,[0,10],x0);

%% Post-processing and plotting
% Calculate wing deflection at the tip (or any other point for that matter)
figure
def_hist = xsol(:,1:n_t)*Lint; % deflection time history
hold on
plot(tsol,def_hist,'LineWidth',1.5)
plot(tsol,0.1*ones(numel(tsol),1),'r','LineWidth',1.5)
hold off
grid on
set(gca,'FontSize',15)
ylabel('\int_0^L \theta(t,x) dx')
xlabel('Time [s]')
legend('y(t)','r(t)')

% Bending displacement
figure
plot(tsol,xsol(:,1:n_t+n_b))
grid on
set(gca,'FontSize',15)
ylabel('Mode amplitude')
xlabel('Time [s]')