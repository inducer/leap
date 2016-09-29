function new_U = RK4(t,U,dt)

% k1 = calc_RK4_chk(t,U);
k1 = op_heat_overset(t,U);
% disp('k1=');disp(k1);

t1 = t + dt/2;
U1 = U + k1.*dt/2;
% k2 = calc_RK4_chk(t1,U1);
k2 = op_heat_overset(t1,U1);
% disp('k2=');disp(k2);

t2 = t + dt/2;
U2 = U + k2.*dt/2;
% k3 = calc_RK4_chk(t2,U2);
k3 = op_heat_overset(t2,U2);
% disp('k3=');disp(k3);

t3 = t + dt;
U3 = U + k3.*dt;
% k4 = calc_RK4_chk(t3,U3);
k4 = op_heat_overset(t3,U3);
new_U = U + (k1+2*k2+2*k3+k4).*dt*1/6;

