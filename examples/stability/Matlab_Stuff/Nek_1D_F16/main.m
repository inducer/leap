% This routine solves the advection equation on two overlapping domains.
% You can adjust a_R to decide the amount of overlap. space_order = 2, 3 
% and 4 uses 1-2-1, 2-4-2 and 3-6-3 SBP first derivative approximation
% respectively.

warning('off','all');
close all
clear variables
clc

global n_L n_R n D_L D_R I c_L c_R M

space_order = 3;
n_L = 72;
n_R = 6;
b_L = 0;
a_L = -1;
tau_L = 1;
tau_R = 1;
t_end = 2;
n = n_L + n_R;
h_L = (b_L-a_L)/(n_L-1);
b_R = 1;
a_R = b_L - h_L*7.5;
h_R = (b_R-a_R)/(n_R-1);
M = zeros(n,n);
P = zeros(n,n);

disp('h_L=');disp(h_L);
disp('h_R=');disp(h_R);

s_L = zeros(n_L,1);
s_R = zeros(n_R,1);
% D_L = zeros(n_L,n_L);
% D_R = zeros(n_R,n_R);
I = zeros(1,n_L);

%%% INITIAL CONDITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fx = zeros(n,1);
x_L = a_L:h_L:b_L;
x_R = a_R:h_R:b_R;
x_plot(1:n_L) = x_L;
x_plot(n_L+1:n) = x_R;

% fx(1:n,1) = gaussmf(x_plot,[0.1 (a_L+b_L)/2]); % for gaussian init condn
start_x = -0.95;
end_x = -0.05;
mid_x = (start_x + end_x)/2;
for j =1:n_L
    if x_L(j) > start_x && x_L(j) < end_x
        fx(j,1) = exp(-1/((x_L(j)-start_x)*(end_x-x_L(j))))/exp(-1/((mid_x-start_x)*(end_x-mid_x)));
    else
        fx(j,1) = 0;
    end
end
for j =1:n_R
    if x_R(j) > start_x && x_R(j) < end_x
        fx(n_L+j,1) = exp(-1/((x_R(j)-start_x)*(end_x-x_R(j))))/exp(-1/((mid_x-start_x)*(end_x-mid_x)));
    else
        fx(n_L+j,1) = 0;
    end
end

v0 = fx;
% plot(x_L,v0(1:n_L,1),x_R,v0(n_L+1:n,1))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% INTERPOLATION VECTOR %%%%%%%%%%%%%%
if space_order==2
    interp_index=1;
    while x_L(interp_index)<a_R
        interp_index = interp_index+1;
    end
    alpha = (a_R - x_L(interp_index-1))/h_L;
    I(1,interp_index-1) = (1-alpha);
    I(1,interp_index) = alpha;
    
elseif space_order==3 || space_order==4
    interp_index=1;
    while x_L(interp_index)<a_R
        interp_index = interp_index+1;
    end
    alpha = (x_L(interp_index) - a_R)/h_L;
    L0 = -1/6*alpha*(1-alpha)*(1+alpha);
    L1 = 1/2*(2-alpha)*alpha*(1+alpha);
    L2 = 1/2*(2-alpha)*(1-alpha)*(1+alpha);
    L3 = -1/6*(2-alpha)*(1-alpha)*alpha;
    I(1,interp_index-2) = L0;
    I(1,interp_index-1) = L1;
    I(1,interp_index) = L2;
    I(1,interp_index+1) = L3;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% OPERATOR CONSTRUCTION %%%%%%%%%%%%%%

[P_L,Q_L,P_R,Q_R] = P_Q_matrices(n_L,n_R,space_order);
P_L = h_L*P_L;
D_L = P_L\Q_L;
P_R = h_R*P_R;
D_R = P_R\Q_R;
P(1:n_L,1:n_L) = P_L;
P(n_L+1:n,n_L+1:n) = P_R;

s_L(1,1) = 1;
s_R(1,1) = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RL = zeros(n_L,n_L);
% jj = interp_index-2;
% RL(jj,jj-2) = -9/586;
% RL(jj,jj-1) = 36/293;
% RL(jj,jj) = -81/293;
% RL(jj,jj+1) = 72/293;
% RL(jj,jj+2) = -45/586;
% 
% jj = interp_index-1;
% RL(jj,jj-2) = -24/173;
% RL(jj,jj-1) = 53/189;
% RL(jj,jj) = -1/112;
% RL(jj,jj+1) = -29/108;
% RL(jj,jj+2) = 30/221;
% 
% jj = interp_index;
% RL(jj,jj-2) = 10/133;
% RL(jj,jj-1) = -29/195;
% RL(jj,jj) = -4/809;
% RL(jj,jj+1) = 57/367;
% RL(jj,jj+2) = -65/846;
% 
% jj = interp_index+1;
% RL(jj,jj-2) = -209/672;
% RL(jj,jj-1) = 209/210;
% RL(jj,jj) = -627/560;
% RL(jj,jj+1) = 209/420;
% RL(jj,jj+2) = -13/209;

%%%%%%%%%% operator_choice = 3 %%%%%%%%%%% 
%%%%%%%%%% For operator_choice=1,2 RL comes from P_Q_matrices file
RL = zeros(n_L,n_L);

% jj = interp_index-3;
% RL(jj+1,jj-1) = -5/1038;
% RL(jj+1,jj) = 20/519;
% RL(jj+1,jj+1) = -15/173;
% RL(jj+1,jj+2) = 40/519;
% RL(jj+1,jj+3) = -25/1038;
% 
% RL(jj+2,jj) = -5/123;
% RL(jj+2,jj+1) = 20/123;
% RL(jj+2,jj+2) = -10/41;
% RL(jj+2,jj+3) = 20/123;
% RL(jj+2,jj+4) = -5/123;
% 
% RL(jj+3,jj+1) = -5/63;
% RL(jj+3,jj+2) = 20/63;
% RL(jj+3,jj+3) = -10/21;
% RL(jj+3,jj+4) = 20/63;
% RL(jj+3,jj+5) = -5/63;
% 
% RL(jj+4,jj+2) = -25/78;
% RL(jj+4,jj+3) = 40/39;
% RL(jj+4,jj+4) = -15/13;
% RL(jj+4,jj+5) = 20/39;
% RL(jj+4,jj+6) = -5/78;
% 
% RL = 1/h_L*RL;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%  PROVING NEGATIVE-SEMIDEFINITENESS  %%%%%%%%
H = P;
% jj = interp_index-3;
% 
% operator_choice = 3;
% if operator_choice==1
%     lmbda = 1343/1512;
%     mu = 71/1512;
%     beta = 1/224;
%     H(jj+1,jj+1) = 3/4*h_L;
%     H(jj+2,jj+2) = 1/3*h_L;
%     H(jj+3,jj+3) = 65/108*h_L;
%     H(jj+4,jj+4) = 5/27*h_L;
% elseif operator_choice==2
%     lmbda = 23/22;
%     mu = 5/22;
%     beta = 9/44;
%     H(jj+1,jj+1) = h_L;
%     H(jj+2,jj+2) = 17/22*h_L;
%     H(jj+3,jj+3) = 1/2*h_L;
%     H(jj+4,jj+4) = 3/11*h_L;
% elseif operator_choice==3
%     lmbda = 61/20;
%     mu = 1/20;
%     beta = 3/4;
%     H(jj+1,jj+1) = 173/60*h_L;
%     H(jj+2,jj+2) = 41/20*h_L;
%     H(jj+3,jj+3) = 21/20*h_L;
%     H(jj+4,jj+4) = 13/60*h_L;
% end    
% hmat = eye(n);
% hmat(1:n_L,1:n_L) = 1/h_L*eye(n_L);
% hmat(n_L+1:n,n_L+1:n) = 1/h_R*eye(n_R);
% H(1:interp_index-3,1:interp_index-3) = lmbda * H(1:interp_index-3,1:interp_index-3);
% H(interp_index+2:n_L,interp_index+2:n_L) = mu * H(interp_index+2:n_L,interp_index+2:n_L);
% H(n_L+1:n,n_L+1:n) = beta * H(n_L+1:n,n_L+1:n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% SYSTEM MATRIX CONSTRUCTION %%%%%%%%%%
c_L = tau_L*(P_L\s_L);      % note that lambda^-1 from H^-1 will cancel lambda in tau_L
c_R = tau_R*(H(n_L+1:n,n_L+1:n)\s_R);

interp_term = -c_R*I;
E_o_n_R_L = zeros(n_R,n_L);
E_o_n_R_L(1,:) = I;
E_I_L = transpose(E_o_n_R_L);

M(1:n_L,1:n_L) = -D_L - c_L*transpose(s_L) + RL;
M(n_L+1:n,n_L+1:n) = -D_R - c_R*transpose(s_R);
M(n_L+1:n,1:n_L) = -interp_term;

% Write system matrix to a file...
dlmwrite('system_matrix_adv',M)

eig(H*M + transpose(H*M));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% SOLVE %%%%%%%%%%%%%%%%%%%

% Pick an integrator: choices are RK, MRAB, or SRAB:
integrator = 'MRAB';
t0 = 0;
cfl = 0.53;
if strcmp(integrator,'MRAB') == 1
    SR = 6;
    dt = SR*cfl*min(h_L,h_R);
else
    dt = cfl*min(h_L,h_R);
end
disp('dt=');disp(dt);
count = 1;
v(1:n,count) = v0;
t(count) = t0;
for time=dt:dt:t_end
    if strcmp(integrator,'RK') == 1
%%% INTEGRATORS: standard Runge-Kutta:
       v0 = RK4(time,v0,dt);
    elseif strcmp(integrator,'SRAB') == 1
%%% INTEGRATORS: singlerate Adams-Bashforth:
        % BOOTSTRAPPING
       if time == dt
         [v0,dv_m2] = RK4_boot(time,v0,dt);
       elseif time == 2*dt  
         [v0,dv_m1] = RK4_boot(time,v0,dt);
       else  
         [v0,dv_m1,dv_m2] = SRAB_3(time,v0,dt,dv_m1,dv_m2);
       end
    elseif strcmp(integrator,'MRAB') == 1
%%% INTEGRATORS: multirate Adams-Bashforth:
    % BOOTSTRAPPING:
       if time == dt
         % Only fill the slow history
         [v0,dv_m2] = RK4_boot(time,v0,dt);
         dR_m2 = dv_m2(n_L+1:n,1);
       elseif time == 2*dt
         % Take some micro-timesteps to fill the fast history
         dt_small = dt/SR;
         for i=1:SR
            [v0,dv_m1] = RK4_boot(time,v0,dt_small);
            if i == SR-1
               dL_m2 = dv_m1(1:n_L,1);
            elseif i == SR
               dL_m1 = dv_m1(1:n_L,1);
            elseif i == 1
               dR_m1 = dv_m1(n_L+1:n,1);
            else
            end
         end
       else  
         % Now we can continue on, using multirate AB.
         [v0,dL_m1,dL_m2,dR_m1,dR_m2] = MRAB_3(time,v0,dt,SR,dL_m1,dL_m2,dR_m1,dR_m2,n_L,n);
       end
    end   
    v(1:n,count) = v0;
    t(count) = time;
    count = count+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%  EIGENVALUE OF THE SYSTEM MATRIX  %%%%%%%%%

e = eig(M);    % Get the eigenvalues of System Matrix
figure;
plot(real(e),imag(e),'r*','markers',10) %   Plot real and imaginary parts
% axis([-4 0 -150 150]);
xlabel('Real','fontsize',20);
ylabel('Imaginary','fontsize',20);
t1 = ('Eigenvalues of system matrix (Adv Eqn)');
set(gca,'fontsize',20);
title(t1,'fontsize',20);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('alpha=');
disp(alpha);

v_plot(1:n_L,1:length(t)) = v(1:n_L,1:length(t));
v_plot(n_L+1:n,1:length(t)) = v(n_L+1:n,1:length(t));

Energy = zeros(length(t),1);    
sys_energy = zeros(length(t),1); 
str = sprintf('%d < X_L < %d    and    %f < X_R < %d', a_L, b_L, a_R, b_R);

start_x_init = start_x;
end_x_init = end_x;
figure;
for i=1:length(t)
    
    %%%%%% EXACT SOLUTION FOR C-1 FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    start_x = start_x_init + t(i);
    end_x = end_x_init + t(i);
    mid_x = (start_x + end_x)/2;
    for j =1:n_L
        if x_L(j) > start_x && x_L(j) < end_x
            exact_sol(j,1) = exp(-1/((x_L(j)-start_x)*(end_x-x_L(j))))/exp(-1/((mid_x-start_x)*(end_x-mid_x)));
        else
            exact_sol(j,1) = 0;
        end
    end
    for j =1:n_R
        if x_R(j) > start_x && x_R(j) < end_x
            exact_sol(n_L+j,1) = exp(-1/((x_R(j)-start_x)*(end_x-x_R(j))))/exp(-1/((mid_x-start_x)*(end_x-mid_x)));
        else
            exact_sol(n_L+j,1) = 0;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    error(1:n,1) = exact_sol(1:n,1) - v(1:n,i);
    Energy(i,1) = transpose(v(:,i))*P*v(:,i);
    sys_energy(i,1) = 2*transpose(v(:,i))*P*M*v(:,i);
%     if i==length(t)/2 || i==(length(t)+1)/2 || i==length(t) || i==1 %|| i==length(t)/4 || i==(length(t)+1)/4 || i==3*length(t)/4 || i==3*(length(t)+1)/4
% %         disp('i=');disp(i);
%         pl_full = plot(x_plot(1:n_L),v_plot(1:n_L,i),'bo',...
%                     x_plot(n_L+1:n),v_plot(n_L+1:n,i),'r+',...
%                     x_L,exact_sol(1:n_L,1),'k:',x_R,exact_sol(n_L+1:n,1),'k:','markers',8);
% %         pl_full = plot(x_plot(1:n_L),error(1:n_L,1),x_plot(n_L+1:n),error(n_L+1:n,1),'r');
% %         axis([a_L b_R 0 1.2]);
%         xlabel('x','fontsize',22);
%         ylabel('w','fontsize',22);
%         legend('left','right')
%         title('Advection Solution');
%         x_dt = get(pl_full,'XData'); 		% Get the plotted data
%         y_dt = get(pl_full,'YData');
%         if i==length(t)/2 || i==(length(t)+1)/2 || i==1
%             imid = find(max(y_dt{1,1}) == y_dt{1,1});
%             text(x_dt{3,1}(imid),y_dt{3,1}(imid)+0.05,[' t = ',num2str(round(t(i)*10)/10)],...
%                 'VerticalAlignment','middle',...
%                 'HorizontalAlignment','left',...
%                 'FontSize',18);
%         elseif i==length(t)
%             imid = find(max(y_dt{2,1}) == y_dt{2,1});
%             text(x_dt{2,1}(imid),y_dt{2,1}(imid)+0.05,[' t = ',num2str(round(t(i)*10)/10)],...
%                 'VerticalAlignment','middle',...
%                 'HorizontalAlignment','left',...
%                 'FontSize',14);
%         end            
%         set(pl_full,'LineWidth',1.3);
%         set(gca,'fontsize',20);
%     end
%     hold on
    pl_full = plot(x_plot(1:n_L),v_plot(1:n_L,i),'bo',x_plot(n_L+1:n),v_plot(n_L+1:n,i),'r+',x_L,exact_sol(1:n_L,1),'k:',x_R,exact_sol(n_L+1:n,1),'k:','markers',8);
    axis([a_L b_R -0.2 1.2]);
    xlabel(str);
    ylabel('V');
    pause(0.001);
    if i<length(t)
        delete(pl_full);
    end
end
box on
hold off

