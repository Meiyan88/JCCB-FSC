function D = updateD(W, group_info)

if nargin == 2
    % for G2,1-norm
    group_set = unique(group_info);
    n_groups = length(group_set);
    for g = 1 : n_groups
        idx = find(group_info == group_set(g));
        d(idx, 1) = 1 ./ sqrt(sum(W(idx, :) .^ 2) + eps);
    end
else
    % for L2,1-norm & L1,1-norm
    d = 1 ./ sqrt(sum(W .^ 2, 2) + eps);
end

D = diag(d);
end