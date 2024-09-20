function query_client_burgers(input_path, buffer)

% ground_fidelity = 128;
% fidelity_list = [16,32,64];
% 
% if m == -1
%     fid = ground_fidelity;
% else
%     fid = fidelity_list(m+1); 
% end

% Re_list = X*500+10;
% nSample = size(X, 1);
%re_List = rand(nSample,1)*500+10;
%g_fid = 128; %fidelity used to calc groud-truth

load(input_path, "X");
load(input_path, "fid");

bs = size(X, 1);

%parameter setting
Paras = {};
Paras.v = 0.002;
Paras.n = fid-1;
Paras.t_n = fid;
Paras.t_end = 1;
% Paras.X = X;
% Paras.a = a;
% Paras.b = b;

% xg = linspace(0,1,fid+1);
% yg = linspace(0,Paras.t_end,fid+1);
% xg_interp = linspace(0,1,interp_dim);
% yg_interp = linspace(0, Paras.t_end,interp_dim);
%% Main

Rec_Y = zeros(Paras.n+1,Paras.t_n+1,bs);
for i = 1:bs
    Paras.a = X(i, 1);
    Paras.b = X(i, 2);
    [Y,time,Time_Ode_solver]=Burger1D_FEM_DBC_SolverF(Paras);
    Rec_Y(:,:,i) = Y;
end

data = {};
data.Y = Rec_Y;
% data.last_Y = Y;

save(buffer, 'data');

pause(3);

end