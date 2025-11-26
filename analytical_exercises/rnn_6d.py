#!/usr/bin/env python3
"""
Problem 6(d) – Minimal RNN trainer for a fixed sequence 101010.

RNN model:
    s_k = w_rec * s_{k-1} + w_x * x_k
    y   = s_T
    L   = (t - y)^2,   where t = number of 1s in the sequence.
"""

import os
import random
from typing import List, Tuple


# ------------------------------------------------------------
# Forward pass: compute all states and final output y
# ------------------------------------------------------------

def forward_rnn(seq: List[int], w_rec: float, w_x: float) -> Tuple[float, List[float]]:
    """
    Forward pass for a single sequence.

    Args:
        seq:   list of bits (0/1), e.g. [1, 0, 1, 0, 1, 0]
        w_rec: recurrent weight
        w_x:   input weight

    Returns:
        y:       final output (s_T)
        states:  list [s_0, s_1, ..., s_T] for use in backprop
    """
    s = 0.0          # initial state s_0
    states = [s]     # store s_0

    for x in seq:
        # recurrence: s_k = w_rec * s_{k-1} + w_x * x_k
        s = w_rec * s + w_x * x
        states.append(s)

    y = s            # final state is the output
    return y, states


# ------------------------------------------------------------
# Backward pass: BPTT for this simple 1-D RNN
# ------------------------------------------------------------

def backward_rnn(
    seq: List[int],
    t: int,
    states: List[float],
    w_rec: float
) -> Tuple[float, float]:
    """
    Backpropagation through time for a single sequence.

    Uses:
        L = (t - y)^2,  y = s_T
        δ_T = dL/ds_T = 2 (y - t)
        δ_k = w_rec * δ_{k+1}
        ∂L/∂w_x   = Σ_k δ_k x_k
        ∂L/∂w_rec = Σ_k δ_k s_{k-1}

    Args:
        seq:    input sequence (list of 0/1)
        t:      target (number of 1s in seq)
        states: list [s_0, ..., s_T] from forward pass
        w_rec:  recurrent weight used in forward

    Returns:
        grad_w_rec, grad_w_x
    """
    T = len(seq)
    y = states[-1]   # s_T

    # delta_T = dL/ds_T
    delta = 2.0 * (y - t)

    grad_w_rec = 0.0
    grad_w_x = 0.0

    # Iterate backwards: k = T-1, ..., 0
    # seq[k] corresponds to time step k+1
    # states[k] is s_k
    for k in reversed(range(T)):
        x_k = seq[k]
        s_prev = states[k]

        # Grad contributions from time step k+1
        grad_w_x   += delta * x_k
        grad_w_rec += delta * s_prev

        # Propagate delta backwards in time: δ_k = w_rec * δ_{k+1}
        delta *= w_rec

    return grad_w_rec, grad_w_x


# ------------------------------------------------------------
# Train the RNN for a single random initialization
# ------------------------------------------------------------

def train_one_run(
    seq: List[int],
    t: int,
    epochs: int,
    eta: float,
    init_seed: int
) -> Tuple[float, float, float, float, float, float]:
    """
    Train the RNN on a single fixed sequence for a given number of epochs.

    Args:
        seq:       fixed input sequence
        t:         target = number of 1s in seq
        epochs:    number of epochs (here: 30)
        eta:       learning rate
        init_seed: seed for random initialization

    Returns:
        (w_rec_init, w_x_init, initial_loss,
         w_rec_final, w_x_final, final_loss)
    """
    rng = random.Random(init_seed)

    # Random initialization of weights in a small range
    w_rec = rng.uniform(-0.5, 0.9)
    w_x   = rng.uniform(-0.5, 0.9)

    # Compute initial loss before any update
    y0, states0 = forward_rnn(seq, w_rec, w_x)
    initial_loss = (t - y0) ** 2
    w_rec_init, w_x_init = w_rec, w_x

    # Training loop: same sequence, repeated for 'epochs' steps
    for _ in range(epochs):
        # Forward pass
        y, states = forward_rnn(seq, w_rec, w_x)
        loss = (t - y) ** 2

        # Backward pass (BPTT)
        grad_wr, grad_wx = backward_rnn(seq, t, states, w_rec)

        # Gradient descent update
        w_rec -= eta * grad_wr
        w_x   -= eta * grad_wx

    # Final loss after training
    y_final, _ = forward_rnn(seq, w_rec, w_x)
    final_loss = (t - y_final) ** 2

    return w_rec_init, w_x_init, initial_loss, w_rec, w_x, final_loss


# ------------------------------------------------------------
# Main: fixed sequence 101010, 30 epochs, save aligned table
# ------------------------------------------------------------

def main() -> None:
    # Fixed input sequence 101010
    seq = [1, 0, 1, 0, 1, 0]
    t = sum(seq)            # target = number of 1s = 3

    epochs = 15
    eta = 0.01              # learning rate (can be adjusted)

    # We will use 3 different random initializations
    init_seeds = [2, 16, 20]

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "rnn_6d_results.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        # General info header
        f.write("Problem 6(d) – RNN counting ones\n")
        f.write(f"Input sequence: {seq}\n")
        f.write(f"Target (number of 1s): {t}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning rate: {eta}\n\n")

        # Table header (aligned columns)
        header = (
            f"{'Run':>3} | "
            f"{'Seed':>4} | "
            f"{'w_rec_init':>12} | "
            f"{'w_x_init':>10} | "
            f"{'initial_loss':>13} | "
            f"{'w_rec_final':>12} | "
            f"{'w_x_final':>10} | "
            f"{'final_loss':>11}\n"
        )
        f.write(header)
        f.write("-" * (len(header) - 1) + "\n")  # separator line (minus final \n)

        # One row per run
        for run_id, seed in enumerate(init_seeds, start=1):
            (
                w_rec_init,
                w_x_init,
                initial_loss,
                w_rec_final,
                w_x_final,
                final_loss,
            ) = train_one_run(
                seq=seq,
                t=t,
                epochs=epochs,
                eta=eta,
                init_seed=seed,
            )

            row = (
                f"{run_id:>3} | "
                f"{seed:>4} | "
                f"{w_rec_init:12.6f} | "
                f"{w_x_init:10.6f} | "
                f"{initial_loss:13.6f} | "
                f"{w_rec_final:12.6f} | "
                f"{w_x_final:10.6f} | "
                f"{final_loss:11.6f}\n"
            )
            f.write(row)

    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
