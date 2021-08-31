function p = predict1(theta, X)
% Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples
p = zeros(m, 1);


y=sigmoid(X*theta);

for something=1:m
    if (y(something)>0.5)
        p(something)=1;
    else
        p(something)=0;
    end
end








% =========================================================================


end
