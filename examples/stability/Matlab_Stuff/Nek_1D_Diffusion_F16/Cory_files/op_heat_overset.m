function dvdt = op_heat_overset(~,v)

global n_L n_R D2_L P_L D_L D2_R P_R D_R E_o_L En sigma_1 sigma_2 M

% M = D2_L + sigma_1*(P_L\(transpose(D_L)))*E_o_L + sigma_2*(P_L\(transpose(D_L)))*En;

dvdt = (M*v);