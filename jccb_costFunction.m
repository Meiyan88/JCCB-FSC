function [J, grad] = jccb_costFunction(theta, X, y, lambda,PX, group_info)
%   Compute cost and gradient for logistic regression with regularization
%   J = costFunction(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the


% Initialize some useful values
m = length(y); 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== Your Code Here  ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial


    h = sigmoid(X*theta);
    D1 = updateD(theta, group_info);
    D3 = updateD(theta);

    J = (((-y)'*log(h)-(1-y)'*log(1-h)))+(lambda(1)*trace(theta'*D1*theta) +lambda(2)*theta'*PX*theta +lambda(3)* sum(abs(theta)));

% calculate grads
    grad = ((X'*(h - y))+lambda(3)*D3*theta +lambda(1)*D1*theta  +2*lambda(2)*PX*theta);



% =============================================================

end
