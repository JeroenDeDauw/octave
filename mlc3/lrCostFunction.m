function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



h = sigmoid(X * theta);
B = (1-y);
C = (1-h);
multi1 =  y .* log(h);
multi2 =  B .* log(C);
J =  sum(multi1 + multi2);
J = J /(-m);
 
max = size(theta);
reg = 0;
for i=2:max
        reg += theta(i) * theta(i);
end
 
reg = reg * (lambda/(2*m));
 
J= J + reg;
 
tam = size(X, 2);
A = (h - y);
 
for i=1:tam
        if i==1,
                grad(i) = ( sum(A .*X(:,i)));
        else
                grad(i) = (( sum(A .*X(:,i))) ) + (lambda * theta(i));
        end
       
        grad(i) = grad(i)/m;
       
end



% =============================================================

grad = grad(:);

end
