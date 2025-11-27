#!/usr/bin/env python3
"""
Problem 6(f) – Bonus-2
RNN trainer for a fixed sequence 101010, using BPTT with a loss at every time step.

RNN model (same recurrence as before):
    s_k = w_rec * s_{k-1} + w_x * x_k

But now:
    • At each time step k, the "desired" output is the cumulative number of ones so far:
          u_k = sum_{i=1}^k x_i
    • We interpret the hidden state as the output at each step, y_k = s_k.
    • The total loss is the average over time:
          L = (1/T) * sum_{k=1}^T (u_k - s_k)^2

We implement BPTT with this loss (see 6(e) and the D2L link).
"""

import os
import random
from typing import List, Tuple


# ------------------------------------------------------------
# Forward pass: compute all states s_0,...,s_T
# ------------------------------------------------------------

def forward_rnn(seq: List[int], w_rec: float, w_x: float) -> List[float]:
    """
    Forward pass for a single sequence.

    Args:
        seq:   list of bits (0/1), e.g. [1, 0, 1, 0, 1, 0]
        w_rec: recurrent weight
        w_x:   input weight

    Returns:
        states: list [s_0, s_1, ..., s_T] for use in backprop
    """
    s = 0.0          # initial state s_0
    states = [s]     # store s_0

    for x in seq:
        # recurrence: s_k = w_rec * s_{k-1} + w_x * x_k
        s = w_rec * s + w_x * x
        states.append(s)

    return states


# ------------------------------------------------------------
# Loss + Backward pass (BPTT with per-time-step loss)
# ------------------------------------------------------------

def loss_and_backward_rnn(
    seq: List[int],
    states: List[float],
    w_rec: float
) -> Tuple[float, float, float]:
    """
    Compute the average loss over all time steps and the gradients using BPTT.

    Model and loss:
        s_k = w_rec * s_{k-1} + w_x * x_k
        u_k = cumulative count of ones up to time k
        L   = (1/T) * sum_{k=1}^T (u_k - s_k)^2

    Let:
        δ_k = dL/ds_k

    Then (scalar version of the D2L formulas):
        local_k = dL_k/ds_k = (2/T) * (s_k - u_k)
        δ_k     = local_k + w_rec * δ_{k+1}
    and
        ∂L/∂w_x   = Σ_k δ_k * x_k
        ∂L/∂w_rec = Σ_k δ_k * s_{k-1}

    Args:
        seq:    input sequence (list of 0/1), length T
        states: list [s_0, ..., s_T] from forward pass
        w_rec:  recurrent weight used in forward

    Returns:
        loss:        average MSE over all time steps
        grad_w_rec:  dL/dw_rec
        grad_w_x:    dL/dw_x
    """
    T = len(seq)
    # Extract s_1,...,s_T; states[0] = s_0
    s_list = states[1:]

    # Compute cumulative targets u_k and loss
    cumulative = 0
    targets = []   # u_k for k=1..T
    loss = 0.0

    for k in range(T):
        cumulative += seq[k]          # u_k = number of ones up to step k+1
        u_k = cumulative
        targets.append(u_k)

        err = s_list[k] - u_k         # s_k - u_k
        loss += err * err

    loss /= T                         # average over time

    # Backward pass (BPTT)
    grad_w_rec = 0.0
    grad_w_x = 0.0
    delta_next = 0.0                  # δ_{T+1} = 0 (no future contribution)

    # Go backwards: k = T-1,...,0   (step index k corresponds to time k+1)
    for k in reversed(range(T)):
        s_k = s_list[k]               # s_{k+1} in time notation
        s_prev = states[k]            # s_k in time notation
        x_k = seq[k]                  # x_{k+1}
        u_k = targets[k]              # target up to step k+1

        # Local gradient of L_k wrt s_k: (2/T)*(s_k - u_k)
        local_grad = (2.0 / T) * (s_k - u_k)

        # Total δ_k = local term + recurrent term from future
        delta = local_grad + w_rec * delta_next

        # Accumulate parameter gradients
        grad_w_x   += delta * x_k
        grad_w_rec += delta * s_prev

        # Pass gradient to previous step
        delta_next = delta

    return loss, grad_w_rec, grad_w_x


# ------------------------------------------------------------
# Train the RNN for a single random initialization (new model)
# ------------------------------------------------------------

def train_one_run(
    seq: List[int],
    epochs: int,
    eta: float,
    init_seed: int
) -> Tuple[float, float, float, float, float, float]:
    """
    Train the RNN on a single fixed sequence for a given number of epochs,
    using the new per-time-step loss and BPTT.

    Args:
        seq:       fixed input sequence
        epochs:    number of epochs (here: 15)
        eta:       learning rate
        init_seed: seed for random initialization

    Returns:
        (w_rec_init, w_x_init, initial_loss,
         w_rec_final, w_x_final, final_loss)
    """
    rng = random.Random(init_seed)

    # Random initialization of weights in a given range
    w_rec = rng.uniform(-0.5, 0.9)
    w_x   = rng.uniform(-0.5, 0.9)

    # Compute initial loss before any update (with new loss definition)
    states0 = forward_rnn(seq, w_rec, w_x)
    initial_loss, _, _ = loss_and_backward_rnn(seq, states0, w_rec)
    w_rec_init, w_x_init = w_rec, w_x

    # Training loop: same sequence, repeated for 'epochs' steps
    for _ in range(epochs):
        # Forward pass
        states = forward_rnn(seq, w_rec, w_x)

        # Loss and gradients via BPTT (per-time-step loss)
        loss, grad_wr, grad_wx = loss_and_backward_rnn(seq, states, w_rec)

        # Gradient descent update
        w_rec -= eta * grad_wr
        w_x   -= eta * grad_wx

    # Final loss after training
    states_final = forward_rnn(seq, w_rec, w_x)
    final_loss, _, _ = loss_and_backward_rnn(seq, states_final, w_rec)

    return w_rec_init, w_x_init, initial_loss, w_rec, w_x, final_loss


# ------------------------------------------------------------
# Main: fixed sequence 101010, 15 epochs, save aligned table
# ------------------------------------------------------------

def main() -> None:
    # Fixed input sequence 101010
    seq = [1, 0, 1, 0, 1, 0]

    epochs = 15
    eta = 0.01              # learning rate

    # 3 different random initializations (as in your version)
    init_seeds = [2, 16, 20]

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "rnn_6f_results.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        # General info header
        f.write("Problem 6(f) – RNN counting ones with BPTT (per-time-step loss)\n")
        f.write(f"Input sequence: {seq}\n")
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
        f.write("-" * (len(header) - 1) + "\n")  # separator line

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
