"""
Train EpOf-QCNDT on the labeled 5G traces.

The circuit is a 4-qubit state vector: Ry(pi x + phi) on each qubit, then
CNOTs on pairs (1,2), (2,3) and (3,4). Pauli-Z expectations are concatenated
with the four scaled features and passed through a tanh layer of 16 units
and a linear layer of 3 class scores.

EpOf maximises training accuracy. The kept model is the one with the best
validation accuracy. Test sessions are not read.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCALED = ROOT / "dataset" / "processed" / "normalized_data" / "selected_features_scaled.csv"
SESSIONS = ROOT / "dataset" / "processed" / "labeled_data" / "session_split.csv"
OUT = ROOT / "results"
FEATURES = ["RSRP", "delta_RSRP", "SNR", "Speed"]
SEED = 42
POPULATION = 50
ITERATIONS = 200
HIDDEN = 16
CLASSES = 3


def statevector_expectations(X, phi):
    """Reference Pauli-Z values from an explicit 16-amplitude simulation."""
    theta = np.pi * X + phi
    state = np.zeros((X.shape[0], 16), dtype=np.complex64)
    state[:, 0] = 1
    for qubit in range(4):
        angle = theta[:, qubit]
        c = np.cos(angle / 2)[:, None]
        s = np.sin(angle / 2)[:, None]
        out = np.empty_like(state)
        step = 1 << qubit
        for base in range(0, 16, step * 2):
            i0 = slice(base, base + step)
            i1 = slice(base + step, base + 2 * step)
            a0 = state[:, i0]
            a1 = state[:, i1]
            out[:, i0] = c * a0 - s * a1
            out[:, i1] = s * a0 + c * a1
        state = out
    for control, target in ((0, 1), (1, 2), (2, 3)):
        out = state.copy()
        c_bit = 1 << control
        t_bit = 1 << target
        for basis in range(16):
            if basis & c_bit and not (basis & t_bit):
                flipped = basis | t_bit
                out[:, basis] = state[:, flipped]
                out[:, flipped] = state[:, basis]
        state = out
    probability = np.abs(state) ** 2
    expectations = np.empty((X.shape[0], 4), dtype=np.float32)
    basis = np.arange(16)
    for qubit in range(4):
        mask = ((basis >> qubit) & 1).astype(np.float32)
        expectations[:, qubit] = 1 - 2 * probability @ mask
    return expectations


def quantum_expectations(X, phi):
    # Same marginals as the state vector after Ry encoding and the CNOT chain.
    theta = np.pi * X + phi
    excited = np.sin(theta / 2) ** 2
    p0 = excited[:, 0]
    p1 = p0 + excited[:, 1] - 2 * p0 * excited[:, 1]
    p2 = p1 + excited[:, 2] - 2 * p1 * excited[:, 2]
    p3 = p2 + excited[:, 3] - 2 * p2 * excited[:, 3]
    return (1 - 2 * np.column_stack([p0, p1, p2, p3])).astype(np.float32)


def forward_scores(X, theta):
    phi = theta[:4]
    hybrid = np.concatenate([X, quantum_expectations(X, phi)], axis=1)
    w1 = theta[4:132].reshape((HIDDEN, 8), order="F")
    b1 = theta[132:148]
    w2 = theta[148:196].reshape((CLASSES, HIDDEN), order="F")
    b2 = theta[196:199]
    hidden = np.tanh(hybrid @ w1.T + b1)
    return hidden @ w2.T + b2


def predict(X, theta):
    return np.argmax(forward_scores(X, theta), axis=1).astype(np.int8)


def accuracy(X, y, theta):
    return float(np.mean(predict(X, theta) == y))


def clip_bounds(theta):
    out = theta.copy()
    out[:4] = np.clip(out[:4], 0.0, 2.0 * np.pi)
    out[4:] = np.clip(out[4:], -1.0, 1.0)
    return out


def levy_step(rng, dim, beta=1.5):
    sigma = (
        math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma, dim)
    v = rng.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta) + 1e-12)


def initialise(rng):
    population = np.empty((POPULATION, 199), dtype=np.float64)
    population[:, :4] = rng.uniform(0, 2 * np.pi, (POPULATION, 4))
    population[:, 4:] = rng.uniform(-1, 1, (POPULATION, 195))
    return population


def propose(rng, position, index, iteration, population, fitness, elite, worst):
    rival_index = int(rng.integers(0, len(population)))
    rival = population[rival_index]
    r2 = rng.random()
    teaching = 1 + int(rng.integers(0, 2))
    direction = 1.0 if rng.random() < 0.5 else -1.0
    if rng.random() < (1 - iteration / ITERATIONS):
        if index < len(population) / 2:
            return position + r2 * (elite - teaching * position) + direction * 0.01 * levy_step(rng, 199)
        brownian = rng.normal(0, 1, 199)
        return elite + np.exp(-iteration / ITERATIONS) * r2 * brownian * (elite - position)
    if fitness[rival_index] > fitness[index]:
        return position + r2 * (1 - iteration / ITERATIONS) * (position - rival) + 0.01 * levy_step(rng, 199)
    return position + r2 * (elite - teaching * rival) + rng.random() * (position - worst)


def per_class(y_true, y_pred):
    rows = []
    for label in range(CLASSES):
        actual = y_true == label
        called = y_pred == label
        tp = int(np.sum(actual & called))
        fp = int(np.sum(~actual & called))
        fn = int(np.sum(actual & ~called))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append((label, precision, recall, int(np.sum(actual))))
    return rows


def check_circuit():
    rng = np.random.default_rng(0)
    X = rng.random((8, 4)).astype(np.float32)
    phi = rng.uniform(0, 2 * np.pi, 4).astype(np.float32)
    got = quantum_expectations(X, phi)
    reference = statevector_expectations(X, phi)
    if np.max(np.abs(got - reference)) > 1e-5:
        raise RuntimeError("closed-form quantum features do not match the state vector")


def main():
    check_circuit()
    frame = pd.read_csv(SCALED)
    sessions = pd.read_csv(SESSIONS)[["session_id", "mobility"]]
    frame = frame.merge(sessions, on="session_id", how="left")
    X = frame[FEATURES].to_numpy(np.float32)
    y = frame["HO_label"].to_numpy(np.int8)
    split = frame["split"].to_numpy()
    mobility = frame["mobility"].to_numpy()

    train = split == "train"
    val = split == "val"
    if np.any(split == "test"):
        pass
    Xtrain, ytrain = X[train], y[train]
    Xval, yval = X[val], y[val]
    majority = float(np.mean(ytrain == np.bincount(ytrain).argmax()))

    rng = np.random.default_rng(SEED)
    population = initialise(rng)
    fitness = np.array([accuracy(Xtrain, ytrain, population[i]) for i in range(POPULATION)])
    order = np.argsort(-fitness)
    population = population[order]
    fitness = fitness[order]

    best_val = -1.0
    best_theta = population[0].copy()
    history = []
    print(f"train {len(ytrain)}  val {len(yval)}  majority {majority:.4f}", flush=True)

    for iteration in range(1, ITERATIONS + 1):
        elite = population[0].copy()
        worst = population[-1].copy()
        for index in range(POPULATION):
            candidate = clip_bounds(
                propose(rng, population[index], index, iteration, population, fitness, elite, worst)
            )
            score = accuracy(Xtrain, ytrain, candidate)
            if score >= fitness[index]:
                population[index] = candidate
                fitness[index] = score
        order = np.argsort(-fitness)
        population = population[order]
        fitness = fitness[order]
        leader_val = accuracy(Xval, yval, population[0])
        if leader_val >= best_val:
            best_val = leader_val
            best_theta = population[0].copy()
        history.append((iteration, float(fitness[0]), float(fitness.mean()), leader_val))
        print(
            f"{iteration:3d}  train {fitness[0]:.4f}  mean {fitness.mean():.4f}  val {leader_val:.4f}",
            flush=True,
        )

    val_pred = predict(Xval, best_theta)
    train_pred = predict(Xtrain, best_theta)
    lines = []
    lines.append("EpOf-QCNDT phase 4")
    lines.append(f"Population {POPULATION}, iterations {ITERATIONS}, seed {SEED}")
    lines.append("Test sessions were not used.")
    lines.append(f"Majority-class training accuracy {majority:.4f}")
    lines.append(f"Selected model training accuracy {accuracy(Xtrain, ytrain, best_theta):.4f}")
    lines.append(f"Selected model validation accuracy {accuracy(Xval, yval, best_theta):.4f}")
    lines.append("Validation by mobility")
    for name in ("Driving", "Static"):
        mask = mobility[val] == name
        if np.any(mask):
            lines.append(f"  {name}: {float(np.mean(val_pred[mask] == yval[mask])):.4f}")
    lines.append("Validation per class (precision, recall, support)")
    for label, precision, recall, support in per_class(yval, val_pred):
        lines.append(f"  class {label}: precision {precision:.4f}, recall {recall:.4f}, support {support}")
    lines.append("Training per class (precision, recall, support)")
    for label, precision, recall, support in per_class(ytrain, train_pred):
        lines.append(f"  class {label}: precision {precision:.4f}, recall {recall:.4f}, support {support}")
    report = "\n".join(lines) + "\n"

    metric_dir = OUT / "metrics"
    log_dir = OUT / "logs"
    fig_dir = OUT / "figures"
    for folder in (metric_dir, log_dir, fig_dir):
        folder.mkdir(parents=True, exist_ok=True)
    (log_dir / "phase4_training.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(history, columns=["iteration", "best_train_accuracy", "mean_train_accuracy", "leader_val_accuracy"]).to_csv(
        metric_dir / "epof_convergence.csv", index=False
    )
    np.savez(metric_dir / "epof_qcndt_parameters.npz", theta=best_theta, features=np.array(FEATURES))
    print(report, flush=True)

    try:
        import matplotlib.pyplot as plt

        table = pd.DataFrame(history, columns=["iteration", "train", "mean", "val"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(table["iteration"], table["train"], label="Best training accuracy")
        ax.plot(table["iteration"], table["val"], label="Leader validation accuracy")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy")
        ax.set_title("EpOf-QCNDT search")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "epof_convergence.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"convergence figure skipped: {exc}")


if __name__ == "__main__":
    main()
