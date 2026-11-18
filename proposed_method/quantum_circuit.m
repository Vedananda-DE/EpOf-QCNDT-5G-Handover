function Z = quantum_circuit(X, phi)
% Pauli-Z expectations of the 4-qubit QCNDT circuit.
% Ry(pi*x + phi) loads each feature, then CNOTs entangle qubits 1-2, 2-3, 3-4.
% These marginals match an explicit 16-amplitude state vector.

theta = pi * X + phi(:).';
excited = sin(theta / 2) .^ 2;
p1 = excited(:, 1);
p2 = p1 + excited(:, 2) - 2 * p1 .* excited(:, 2);
p3 = p2 + excited(:, 3) - 2 * p2 .* excited(:, 3);
p4 = p3 + excited(:, 4) - 2 * p3 .* excited(:, 4);
Z = 1 - 2 * [p1, p2, p3, p4];
end
