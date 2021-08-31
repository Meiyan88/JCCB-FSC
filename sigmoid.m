function g = sigmoid(z)
%   Compute sigmoid functoon
%   g = sigmoid(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== Your Code Here ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = (exp(-z)+1).^-1;



% =============================================================

end
