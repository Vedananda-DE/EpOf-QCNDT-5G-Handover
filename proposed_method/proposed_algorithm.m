function [scores, hybrid] = proposed_algorithm(X, theta)
% EpOf-QCNDT forward pass.
% theta has 199 entries: 4 quantum angles, then the 8-16-3 layer.
% The hidden layer uses tanh. The three outputs are class scores.

Z = quantum_circuit(X, theta(1:4));
hybrid = [X, Z];

W1 = reshape(theta(5:132), 16, 8);
b1 = theta(133:148).';
W2 = reshape(theta(149:196), 3, 16);
b2 = theta(197:199).';

hidden = tanh(hybrid * W1.' + b1);
scores = hidden * W2.' + b2;
end
