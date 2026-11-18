# Phase 4: EpOf-QCNDT

The classifier uses the four selected features `RSRP`, `delta_RSRP`, `SNR` and `Speed`, already scaled to [0, 1] on the training sessions.

## Circuit

Each feature is loaded onto one qubit by `Ry(pi x)`. A trainable angle `phi` in `[0, 2 pi]` is applied with a second `Ry` on the same qubit. CNOTs then entangle qubit pairs (1,2), (2,3) and (3,4). The four Pauli-Z expectations are concatenated with the four classical features, giving the 8-dimensional hybrid vector in the paper.

The fully connected head is 8 inputs, 16 tanh units, and 3 class scores. The 199 search variables are the 4 angles, the 8-by-16 weights, 16 biases, the 16-by-3 weights, and 3 biases. Classical weights stay in `[-1, 1]`.

## Optimizer

EpOf keeps 50 candidates for 200 iterations. Fitness is training accuracy. Early iterations explore: fitter members move by the zebra foraging step with a small Levy perturbation, and the others follow the leader with Brownian noise. Later iterations exploit: a member that loses to a rival steps away from that rival, and a member that is fitter moves toward the leader and away from the worst candidate. A new position is kept only when training accuracy does not fall. The model that is saved is the leader with the highest validation accuracy.

The test split is not used in this phase.

The hidden-unit nonlinearity is tanh. The paper specifies the layer sizes and the softmax decision, and it does not name the hidden activation.
