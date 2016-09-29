function dvdt = P_Q_43_overset(t,v)

global n_L n_R n D_L D_R I c_L c_R M

% D = zeros(n,n);
% d = zeros(n,1);
% 
% D(1:n_L,1:n_L) = D_L;
% D(n_L+1:n,n_L+1:n) = D_R;
% D(n_L+1:n,1:n_L) = -c_R*I;
% d(1:n_L,1) = v(1,1)*c_L;
% d(n_L+1:n,1) = v(n_L+1,1)*c_R;

dvdt = M*v;%-(D*v) - d;