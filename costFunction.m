#! /usr/bin/octave

# trainingMatrix (X) The training matrix with examples, first column being a one-filled row vector.
# trainingVector(y) The expected results, row vector with same row count as trainingMatrix.
# hypothesis (theta) The 'solution' hypothesis, vector with same length as the column count of trainingMatrix.
function J = costFunction( trainingMatrix, trainingVector, hypothesis )

tariningSetSize = size( trainingMatrix, 1 );
predictions = trainingMatrix * hypothesis;

J = sum( ( predictions - trainingVector ) .^ 2 )  / ( 2 * tariningSetSize );