function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha


m = length(y); % number of training examples
n = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
	
	costs = zeros(n);
	
	for i = 1:m
		cost = 0;
		
		for t = 1:n
			cost = cost + theta(t) * X(i, t);
		end
		
		cost = cost - y(i);
		
		for t = 1:n
			costs(t) = costs(t) + cost * X(i, t);
		end
	end
	
	for t = 1:n
		theta(t) = theta(t) - alpha / m * costs(t);
	end
	  
	J_history(iter) = computeCost(X, y, theta);

end

end
