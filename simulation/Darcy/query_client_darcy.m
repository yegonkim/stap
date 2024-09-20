function query_client_darcy(input_path, buffer)

load(input_path, "X");
% load(input_path, "fid");

bs = size(X, 1);

%parameter setting
s = size(X,2);
f = ones(s,s);
%% Main

Rec_Y = [];
for i = 1:bs
    % Y0 = zeros(s,s);
    Y0 = X(i, :, :);
    Y0 = reshape(Y0, [s, s]);
    Y = solve_gwf(Y0, f);
    Rec_Y(i, :, :) = Y;
end

data = {};
data.Y = Rec_Y;

save(buffer, 'data');

pause(3);

end
