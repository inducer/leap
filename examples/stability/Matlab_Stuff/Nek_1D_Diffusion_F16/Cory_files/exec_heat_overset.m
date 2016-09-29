close all
clear all
% clc

global n_L n_R D2_L P_L D_L D2_R P_R D_R E_o_L E_n_L sigma_1 sigma_2 M

bndry_order = 1;
n_pts = 40;
n_L = n_pts;
n_R = 3/2*n_pts;
b_L = 0;
a_L = -1;

t_end = 1;

n = n_L + n_R;
h_L = (b_L-a_L)/(n_L-1);
b_R = 1 - h_L*0.2;
a_R = b_L - h_L*0.2;
h_R = (b_R-a_R)/(n_R-1);
x_L = a_L:h_L:b_L;
x_R = a_R:h_R:b_R;
a = 1.0;
epsilon = 0.01;


M = zeros(n,n);
Ma = zeros(n,n);
Md = zeros(n,n);
P = zeros(n,n);
P_tilde = zeros(n,n);
I_L = zeros(1,n_L);
I_R = zeros(1,n_R);

%%% Construction of Interpolation Matrices
interp_index_L=1;
while x_L(interp_index_L)<a_R
    interp_index_L = interp_index_L+1;
end
interp_index_R=1;
while x_R(interp_index_R)<=b_L
    interp_index_R = interp_index_R+1;
end
disp('interp_index_L=');disp(interp_index_L);
disp('interp_index_R=');disp(interp_index_R);

dist_L = (a_R - x_L(interp_index_L-1))/h_L;
dist_R = (b_L - x_R(interp_index_R-1))/h_R;

alpha_L = 1 - dist_L;
alpha_R = 1 - dist_R;

I_L(1,interp_index_L-1) = alpha_L;
I_L(1,interp_index_L) = 1-alpha_L;

I_R(1,interp_index_R-1) = alpha_R;
I_R(1,interp_index_R) = 1-alpha_R;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% P,BS,D & D2 matrices %%%%%%%%%%%%%%%%%%
[P_L,Q_L,S_L,D2_L,P_R,Q_R,S_R,D2_R] = P_Q_heat_overset(n_L,n_R,bndry_order,alpha_L,alpha_R);
P_L = h_L*P_L;
D_L = P_L\Q_L;
D2_L = 1/(h_L*h_L)*D2_L;
P_R = h_R*P_R;
D_R = P_R\Q_R;
D2_R = 1/(h_R*h_R)*D2_R;
S_L = 1/h_L*S_L;
S_R = 1/h_R*S_R;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% Initial Condition %%%%%%%%%%%%%%%%%%%%
fx = zeros(n,1);
x(1:n_L) = x_L; 
x(n_L+1:n) = x_R;
% fx(1:n,1) = gaussmf(x,[0.1 (b_R+a_L)/2]);
% fx(1:n,1) = sin (pi*x);

sigma = 50;
lambda = -1;
del = 1/(4*epsilon*sigma);
for j =1:n
%     fx(j,1) = sin((x(j)+1)*pi/2);
    fx(j,1) = exp(-((x(j)+0.5)^2)/(4*epsilon*del));
end
v0 = fx;
% plot(x,fx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Construction of E Matrices
E_o_L = zeros(n_L,n_L);
E_n_L = zeros(n_L,n_L);
E_n_o_L_R = zeros(n_L,n_R);
E_I_L = zeros(n_L,n_R);
E_o_L(1,1) = 1;
E_n_L(n_L,n_L) = 1;
E_n_o_L_R(n_L,:) = I_R;
% E_n_o_L_R(n_L,1) = I_R(1,interp_index_R-1);
% E_n_o_L_R(n_L,2) = I_R(1,interp_index_R);

E_o_R = zeros(n_R,n_R);
E_n_R = zeros(n_R,n_R);
E_o_n_R_L = zeros(n_R,n_L);
E_o_R(1,1) = 1;
E_n_R(n_R,n_R) = 1;
E_o_n_R_L(1,:) = I_L;
% E_o_n_R_L(1,n_L-1) = I_L(1,interp_index_L-1);
% E_o_n_R_L(1,n_L) = I_L(1,interp_index_L);

E_I_L = transpose(E_o_n_R_L);
E_I_R = transpose(E_n_o_L_R);

% disp('E_o_n_R_L=');disp(E_o_n_R_L);
% disp('E_n_o_L_R=');disp(E_n_o_L_R);
% disp('E_I_L=');disp(E_I_L);
% disp('E_I_R=');disp(E_I_R);

s_L = zeros(n_L,1);
s_R = zeros(n_R,1);
s_L(1,1) = 1;
s_R(1,1) = 1;
tau_L = 1;
tau_R = 1/2;

c_L = tau_L*(P_L\s_L);
c_R = tau_R*(P_R\s_R);
interp_term = -c_R*I_L;

sigma_1 = 1;
sigma_2 = -1;
% sigma_3 = C_R(1,1)*2*beta_R;
tau_2 = -1;
tau_3 = -1;
% tau_1 = - a/2 - ((-epsilon_tau_2+tau_3)^2)/(2*epsilon*k_R);

%%% System Matrix
Ma(1:n_L,1:n_L) = a*(- D_L - c_L*transpose(s_L) + tau_R*(P_L\(transpose(I_L)))*I_L); 
Ma(1:n_L,n_L+1:n) = a*(-tau_R*(P_L\E_I_L));
Ma(n_L+1:n,n_L+1:n) = a*(-D_R - c_R*transpose(s_R));
Ma(n_L+1:n,1:n_L) = a*(-interp_term);

Md(1:n_L,1:n_L) = epsilon*(D2_L + sigma_1*(P_L\(transpose(S_L)))*E_o_L + sigma_2*(P_L\(transpose(S_L)))*E_n_L);
Md(1:n_L,n_L+1:n) = epsilon*(-sigma_2*(P_L\(transpose(S_L)))*E_n_o_L_R);
Md(n_L+1:n,n_L+1:n) = epsilon*(D2_R + tau_2*(P_R\(transpose(S_R)))*E_n_R - tau_3*(P_R\transpose(I_R))*I_R*S_R);
Md(n_L+1:n,1:n_L) = epsilon*(tau_3*(P_R\E_I_R)*S_L);

% M(1:n_L,1:n_L) = a*(- D_L - c_L*transpose(s_L) + tau_R*(P_L\(transpose(I_L)))*I_L) + epsilon*(D2_L + sigma_1*(P_L\(transpose(S_L)))*E_o_L + sigma_2*(P_L\(transpose(S_L)))*E_n_L);
% M(1:n_L,n_L+1:n) = a*(-tau_R*(P_L\E_I_L)) + epsilon*(-sigma_2*(P_L\(transpose(S_L)))*E_n_o_L_R);
% M(n_L+1:n,n_L+1:n) = a*(-D_R - c_R*transpose(s_R)) + epsilon*(D2_R + tau_2*(P_R\(transpose(S_R)))*E_n_R - tau_3*(P_R\transpose(I_R))*I_R*S_R);
% M(n_L+1:n,1:n_L) = a*(-interp_term) + epsilon*(tau_3*(P_R\E_I_R)*S_L);

M = Ma + Md;
P(1:n_L,1:n_L) = P_L;
P(n_L+1:n,n_L+1:n) = P_R;
H = eye(n);
H(1:n_L,1:n_L) = h_L*H(1:n_L,1:n_L);
H(n_L+1:n,n_L+1:n) = h_R*H(n_L+1:n,n_L+1:n);

t0 = 0;
%%% Solve
% tspan = [t0 t_end];
% [t, v] = ode45('op_heat_overset',tspan,v0);
% v=transpose(v);

% % v=v0;
cfl = 0.25;
dt = cfl*min(h_L,h_R)*min(h_L,h_R);
count = 1;
v(1:n,count) = v0;
t(count) = t0;
for time=dt:dt:t_end
%         disp('t='),disp(t);
    v0 = RK4(time,v0,dt);
    v(1:n,count) = v0;
    t(count) = time;
    count = count+1;
end

%%% Eigenvalue plotting
e = eig(M);    % Get the eigenvalues of System Matrix
figure;
p_eig = plot(real(e),imag(e),'r*'); %   Plot real and imaginary parts
xlabel('Real');
ylabel('Imaginary');
t1 = ('Eigenvalues of system matrix (1)');
title(t1);
% saveas(p_eig,'sys_matrix.png');

%%% Post-processing
exact_sol = zeros(n,1);
sys_energy = zeros(length(t),1);
energy = zeros(length(t),1); 
figure();
for i=1:length(t)
% %     disp(i);
% %     exact_sol(1:n,1) = exp(-(pi^2)*t(i))*sin(pi*x);
%     for j =1:n
%         exact_sol(j,1) = sqrt(del/(t(i)+del))*exp(-((x(j)+lambda*t(i)+0.5)^2)/(4*epsilon*(t(i)+del)));
%     end
%     energy(i,1) = transpose(v(:,i))*P*v(:,i);
%     sys_energy(i,1) = 2*transpose(v(:,i))*P*M*v(:,i);
%     error(1:n,i) = exact_sol(1:n,1) - v(1:n,i);
%     if i==length(t)/2 || i==(length(t)+1)/2 || i==length(t)|| i==1
%         pl_full = plot(x(1:n_L),v(1:n_L,i),'bo',x(n_L+1:n),v(n_L+1:n,i),'r+',x,exact_sol(1:n,1),'k:');
%     %         pl_full = plot(x,error(1:n,i));
%         axis([-1 1 0 1.2]);
%         xlabel('x');
%         ylabel('V');
%         title('Solution at different times');
% %     x_dt = get(pl_full,'XData'); 		% Get the plotted data
% %     y_dt = get(pl_full,'YData');
% %     imid = find(max(y_dt{3,1}) == y_dt{3,1});
% %     text(x_dt{3,1}(imid),y_dt{3,1}(imid)+0.05,[' t = ',num2str(round(t(i)*100)/100)],...
% %         'VerticalAlignment','middle',...
% %         'HorizontalAlignment','left',...
% %         'FontSize',14);
% %     set(pl_full,'LineWidth',2);
% %     set(gca,'fontsize',20);
% % %         text(0.6,1.1,['Time = ',num2str(t(i,1))],'FontSize',15);
%     end
%     hold on
    if (rem(i,20)==0)
        pl_full = plot(x(1:n_L),v(1:n_L,i),'bo',x(n_L+1:n),v(n_L+1:n,i),'r+',x(1:n_L),exact_sol(1:n_L,1),'k:',x(n_L+1:n),exact_sol(n_L+1:n,1),'k:','markers',8);
        axis([a_L b_R -0.2 1.2]);
        xlabel('x');
        ylabel('V');
        pause(0.001);
        if i<length(t)
            delete(pl_full);
        end
    end
end
hold off
% figure();
% p2 = plot(t,energy);
% xlabel('Time','fontsize',20);
% ylabel('Energy( ||u|| _P^2 )','fontsize',20);
% title('Scheme-1','fontsize',15);
% set(p2,'LineWidth',2);
% set(gca,'fontsize',20)
% % saveas(p2,'Energy_new.png');
% 
% figure();
% p1 = plot(t,sys_energy);
% xlabel('Time','fontsize',20);
% ylabel('Energy Rate ( d/dt ||u|| _P^2 )','fontsize',20);
% title('Scheme-1','fontsize',15);
% set(p1,'LineWidth',2);
% set(gca,'fontsize',20)
% saveas(p1,'Energy_rate_new.png');