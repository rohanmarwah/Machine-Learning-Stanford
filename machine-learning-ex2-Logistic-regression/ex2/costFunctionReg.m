function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta)-1;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for i=1:m
  J=J-(y(i)*log(sigmoid(X(i,:)*theta))+(1-y(i))*log(1-sigmoid(X(i,:)*theta)))/m ;
endfor 
for j=2:n+1
  J=J+lambda*theta(j)^2/(2*m);
endfor

for j=1:n+1
for i=1:m
  grad(j)=grad(j)+(sigmoid(X(i,:)*theta)-y(i))*X(i,j)/m;
endfor
endfor
for j=2:n+1
   grad(j)=grad(j)+ lambda*theta(j)/m;
 endfor







% =============================================================

end
