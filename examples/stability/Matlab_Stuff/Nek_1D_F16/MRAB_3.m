function [new_U,dL_m1,dL_m2,dR_m1,dR_m2] = MRAB_3(t,U,dt,SR,dL_m1,dL_m2,dR_m1,dR_m2,n_L,n)

% A fastest-first scheme specifically designed for this problem

% Fast timestep:

dt_L = dt/SR;
U_L = U(1:n_L,1);
U_R = U(n_L+1:n,1);

% March the left (fast) solution, updating these components in the
% U-vector using the proper scheme, whilst extrapolating to get the right
% (slow) solution.

dt_r_ext = dt_L;

du = P_Q_43_overset(t,U);
% Split out left and right parts:
du_L = du(1:n_L,1);
du_R = du(n_L+1:n,1);

for i=1:SR
   new_U_L = U_L + (23/12)*dt_L*du_L - (4/3)*dt_L*dL_m1 + (5/12)*dt_L*dL_m2;
   new_U_R = U_R + (23/12)*dt_r_ext*du_R - (4/3)*dt_r_ext*dR_m1 + (5/12)*dt_r_ext*dR_m2;
   % Update fast histories and timestep for slow extrapolation
   dL_m2 = dL_m1;
   dL_m1 = du_L;
   dt_r_ext = dt_r_ext + dt_L;
   % Recombine left and right components for next call to overset routine.
   U(1:n_L,1) = new_U_L;
   U(n_L+1:n) = new_U_R;
   % Update time.
   t = t + dt_L;
   % Recall overset routine for next RHS values.
   du = P_Q_43_overset(t,U);
   % Only use fast part of resulting RHS/state update in next step.
   du_L = du(1:n_L,1);
   U_L = new_U_L;
end

% Then, update the slow history values and the outputs at the end of a
% macrostep:

dR_m2 = dR_m1;
dR_m1 = du_R;

new_U(1:n_L,1) = new_U_L;
new_U(n_L+1:n,1) = new_U_R;
