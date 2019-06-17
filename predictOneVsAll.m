function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

%set probability for a given number of num_labels
prob = zeros(num_labels,1);

%predict for only for m number of images
%in each prediction, predict against all 10 labels
%pick the one with the highest probabiliy
%eg: given an image with 401 cols, and all_theta has 2 rows of 401 cols,
%find prob for row 1 => X * all_theta(1) and
%find prob for row 2 => X * all_theta(2)
%pick the row with the highest probability.(if row two, then the predicted
%number is 2).

%note: I haven't fully used a vectorized approach
%I can replace my for loop with a vectorized approach
 for i=1:1:m    
   %eg: X(i,:) will be 1 by 401 row vector
   %all_theta : is 10 * 401. all_theta' => 401 * 10
   %to prevent an error, find the transpose of all_theta to be 401 by 10
   %so their product will give 1 by 10 row vect
   % 1 by 401 * 401 by 10 => 1 by 10
   %their transpose gives a 10 by 1 col vect
   prob = (sigmoid(X(i,:) * all_theta'))';  
  
   %the index with the hightest probability gives the predicted value of
   %that digit X(i,image)
   %find the max value of the array and its index(which gives its value)
   %note that index 10 here represents the value 0;
   [Maxi,index] = max(prob);
   
   
   %the found index gives the predicted value
   p(i) = index;
   
   %clear after each iter
   Maxi = 0; index = -1;
 end
 
% =========================================================================
end


