function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients fore
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%feedforward

y_bin=zeros(num_labels,m);
Jvec=zeros(num_labels,1);
for i=1:m
	A1=X(i,:);				 % a1 is 1x400
	Z2=Theta1*[1;A1'];       % theta1*a1'  (25x401)x(401x1)
	A2=1./(1+exp(-Z2));      % a2 is 25x1 vector because z2 is 25x1
	Z3=Theta2*[1;A2];        % z3 is 10x1
	A3=1./(1+exp(-Z3));      % a3 is h(X) which is 10x1

	c=y(i);
	y_bin(:,i)=zeros(num_labels,1);		% converting y into binary 
	y_bin(c,i)=1;				% y(i) is 10x1

	Jvec=Jvec-(1/m).*(y_bin(:,i).*log(A3)+((1-y_bin(:,i)).*(log(1-A3))));

end

	J = sum(Jvec)  + (lambda/(2*m))*(sum(sum((Theta1(:,2:end)).^2)) + sum(sum((Theta2(:,2:end)).^2)))   ;



%Backpropogation
delta_3=zeros(num_labels,1);
delta_2=zeros(hidden_layer_size,1);
capital_delta1=zeros(hidden_layer_size,input_layer_size+1);
capital_delta2=zeros(num_labels,hidden_layer_size+1);

for i=1:m
	A1=([1 ,X(i,:)])';				 % a1 is 1x400
	Z2=Theta1*[A1];     	 % theta*a1'  (25x401)x(401x1)
	A2=1./(1+exp(-Z2));
	A2=[1;A2];      % a2 is 25x1 vector because z2 is 25x1
	Z3=Theta2*[A2];        % z3 is 10x1
	A3=1./(1+exp(-Z3));      % a3 is h(X) which is 10x1

	c=y(i);
	y_bin(:,i)=zeros(num_labels,1);		% converting y into binary 
	y_bin(c,i)=1;						% y(i) is 10x1

	delta_3=A3-y_bin(:,i);											%delta3 is (number of labels)x1 vector
	delta_2=(((Theta2(:,2:end))')*delta_3).*sigmoidGradient(Z2);
																	%delta2 is (hidden layer size)x1 vector

	capital_delta1=capital_delta1+delta_2*(A1');					%so captal delta 1 is hidden layer size x input layer size
	capital_delta2=capital_delta2+delta_3*(A2');					%A2 is 25x1 and A1 is 1x400

end



% regularised gradient
 Theta1_grad(:,2:end)=capital_delta1(:,2:end)/m +(lambda*Theta1(:,2:end))/m;
 Theta1_grad(:,1)=capital_delta1(:,1)/m;

Theta2_grad(:,2:end)=capital_delta2(:,2:end)/m  +(lambda*Theta2(:,2:end))/m;
 Theta2_grad(:,1)=capital_delta2(:,1)/m;

 




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
