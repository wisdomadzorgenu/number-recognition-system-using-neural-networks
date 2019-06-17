function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

% Add ones to the X data matrix
X = [ones(m, 1) X];

%fprintf('\n the size of X is '); size(X)
%fprintf('\n the size of theta1 is '); size(Theta1)

 %first activation in the hidden layer corresponds to
 % the g(product of xes and theta1
 first_activation = zeros(1,size(Theta1,1));
 output = zeros(size(Theta2,1),1); %tends to be our output

%note: I haven't fully used a vectorized approach
%I can replace my for loop with a vectorized approach
 for i=1:1:m    
   %compute the second layer(the hidden layer) using vectorization approach
   %eg: X(i,:) will be 1 by 401 row vector
   %theta1 : is 25 * 401. theta1' => 401 * 25
   %to prevent an error, find the transpose of Theta1 to be 401 by 25
   %so their product will give 1 by 25 row vect
   % 1 by 401 * 401 by 25 => 1 by 25
   %their transpose gives a 25 by 1 col vect
   first_activation = sigmoid(X(i,:) * Theta1');   
    
   %compute values for outupt layer
   %since Theta2 has a size of 10 by 26,
   %we're adding ones to as another first col to make our first activations
   %1 by 26
   %fprintf('\n Size of first_activ before adding ones is')
   %size(first_activation)
   
   %add one to first column
   first_activation = [1 first_activation];
   
   %fprintf('\n Size of first_activ after adding ones is')
   %size(first_activation)
   
   %so first_activation now becomes 1 by 26
   % 1 by 26 * 10 by 26 isn't possible
   % so we're going to find the transpose of theta2 => 26 by 10
   % 1 by 26 * 26 by 10 => 1 by 10 :hypothesis
   %fprintf('\ntheta2 has a size of ');size(Theta2')
   output = ( sigmoid(first_activation * Theta2') )';  

   %the index with the hightest probability gives the predicted value of
   %that digit X(i,image)
   %find the max value of the array and its index(which gives its value)
   %note that index 10 here represents the value 0;
   [Maxi,index] = max(output);
   
   
   %the found index gives the predicted value
   p(i) = index;
   
   %clear after each iter
   Maxi = 0; index = -1;
 end

% =========================================================================
end
