function J = costFunction( trainingMatrix, trainingVector, theta )

tariningSetSize = size( trainingMatrix, 1 );
predictions = trainingMatrix * theta;

J = sum( ( predictions - trainingVector ) .^ 2 )  / ( 2 * tariningSetSize );