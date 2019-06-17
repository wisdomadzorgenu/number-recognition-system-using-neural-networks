function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

   %intuition: g(z) = 1 / (1 + e^-z)
  g = 1.0 ./ (1.0 + exp(-z));
end
