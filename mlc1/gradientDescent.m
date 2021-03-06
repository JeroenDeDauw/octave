function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha


exampleCount = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

	cost = ( X * theta - y )';
	
	for featureIndex = 1:length(theta)
		theta(featureIndex) = theta(featureIndex) - alpha / exampleCount * ( cost * X(:,featureIndex) );
	end
	  
	J_history(iter) = computeCost(X, y, theta);

end

end
