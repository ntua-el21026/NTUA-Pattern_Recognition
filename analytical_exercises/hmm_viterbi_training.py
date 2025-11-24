#!/usr/bin/env python3
"""
Minimal Viterbi-training script for Problem 2(c).

What it does:
1. Defines the HMM of Problem 2:
   - States: 1, 2, 3
   - Observations: H, T
   - Given initial pi, A, B.
2. Uses the training sequence:
   HHTTTHHTTTTTHHHH
3. Runs several epochs of Viterbi-training:
   (E-step: Viterbi, M-step: frequency re-estimation).
4. After EACH epoch, logs:
   - best Viterbi path
   - joint probability of path and observations
   - updated pi, A, B
5. Writes everything into ONE file:
   results/hmm_viterbi_training_log.txt
"""

import os


def viterbi(pi, A, B, obs_seq, obs_map, state_labels):
    """
    Run the Viterbi algorithm for a single sequence.

    Parameters
    ----------
    pi : list[float]
        Initial distribution, length N.
    A : list[list[float]]
        Transition matrix A[i][j] = P(q_t = j | q_{t-1} = i).
        Shape N x N.
    B : list[list[float]]
        Emission matrix B[j][k] = P(O_t = obs_k | q_t = j).
        Shape N x M, where M = number of observation symbols.
    obs_seq : list[str]
        Observation sequence (e.g. ['H','H','T',...]).
    obs_map : dict[str, int]
        Maps each observation symbol to a column index in B.
    state_labels : list[int]
        Human-readable labels for states (e.g. [1,2,3]).

    Returns
    -------
    best_path_labels : list[int]
        Most likely state sequence (MAP path) in terms of state labels.
    best_prob : float
        Joint probability of that path with the observations.
    """
    T = len(obs_seq)             # length of observation sequence
    N = len(state_labels)        # number of states

    # delta[t][j] = max probability of any path ending in state j at time t
    # psi[t][j]   = argmax previous state for that path (backpointer)
    delta = [[0.0] * N for _ in range(T)]
    psi = [[-1] * N for _ in range(T)]

    # ---------- Initialization (t = 0) ----------
    o0 = obs_map[obs_seq[0]]
    for j in range(N):
        delta[0][j] = pi[j] * B[j][o0]
        psi[0][j] = -1  # no predecessor at t = 0

    # ---------- Recursion (t = 1, ..., T-1) ----------
    for t in range(1, T):
        ot = obs_map[obs_seq[t]]
        for j in range(N):
            # For each destination state j, find best previous state i
            best_val = -1.0
            best_i = -1
            for i in range(N):
                val = delta[t - 1][i] * A[i][j]
                if val > best_val:
                    best_val = val
                    best_i = i
            # Multiply by emission probability for observation at time t
            delta[t][j] = best_val * B[j][ot]
            psi[t][j] = best_i

    # ---------- Termination ----------
    last_t = T - 1
    best_last_state = max(range(N), key=lambda j: delta[last_t][j])
    best_prob = delta[last_t][best_last_state]

    # ---------- Backtracking ----------
    best_path_indices = [best_last_state] * T
    for t in range(last_t, 0, -1):
        best_path_indices[t - 1] = psi[t][best_path_indices[t]]

    # Convert internal indices (0..N-1) to external labels (1..N)
    best_path_labels = [state_labels[idx] for idx in best_path_indices]
    return best_path_labels, best_prob


def reestimate_parameters(path_labels, obs_seq, obs_map,
                          num_states, num_obs,
                          pi_old, A_old, B_old):
    """
    One M-step of Viterbi-training (hard EM).

    Given:
      - path_labels: Viterbi-decoded state sequence (labels like 1,2,3),
      - obs_seq: observation sequence (e.g. ['H','T',...]),
    treat them as fully observed and compute ML estimates:

      pi_new[i]   ~ frequency of starting in state i
      A_new[i][j] ~ frequency of i->j transitions (normalized over j)
      B_new[j][k] ~ frequency of symbol k in state j (normalized over k)

    If a state never appears in the path (no counts),
    we fall back to the OLD parameters for that state
    to avoid degenerate rows.

    Parameters
    ----------
    path_labels : list[int]
        State labels (1..num_states) for each time step.
    obs_seq : list[str]
        Observation sequence.
    obs_map : dict[str, int]
        Maps observation symbol -> index.
    num_states : int
        Number of states.
    num_obs : int
        Number of distinct observation symbols.
    pi_old, A_old, B_old :
        Old parameters (used as fallback if there are no counts).

    Returns
    -------
    pi_new : list[float]
    A_new  : list[list[float]]
    B_new  : list[list[float]]
    """
    T = len(obs_seq)

    # ---------- Initial distribution pi ----------
    pi_new = [0.0] * num_states
    start_state_idx = path_labels[0] - 1  # convert label (1..N) to index (0..N-1)
    pi_new[start_state_idx] = 1.0

    # ---------- Transition counts ----------
    trans_counts = [[0] * num_states for _ in range(num_states)]
    for t in range(1, T):
        i = path_labels[t - 1] - 1  # from state index
        j = path_labels[t] - 1      # to state index
        trans_counts[i][j] += 1

    # Normalize counts -> A_new.
    # If a state i never appears (row_sum == 0), keep A_old[i].
    A_new = [[0.0] * num_states for _ in range(num_states)]
    for i in range(num_states):
        row_sum = sum(trans_counts[i])
        if row_sum > 0:
            A_new[i] = [c / row_sum for c in trans_counts[i]]
        else:
            # No transitions from state i observed: keep old row.
            A_new[i] = A_old[i][:]

    # ---------- Emission counts ----------
    emit_counts = [[0] * num_obs for _ in range(num_states)]
    state_visit_counts = [0] * num_states
    for t, label in enumerate(path_labels):
        s_idx = label - 1          # state index
        o_idx = obs_map[obs_seq[t]]  # observation index
        emit_counts[s_idx][o_idx] += 1
        state_visit_counts[s_idx] += 1

    # Normalize counts -> B_new.
    # If a state j is never visited, keep B_old[j].
    B_new = [[0.0] * num_obs for _ in range(num_states)]
    for j in range(num_states):
        total = state_visit_counts[j]
        if total > 0:
            B_new[j] = [c / total for c in emit_counts[j]]
        else:
            B_new[j] = B_old[j][:]

    return pi_new, A_new, B_new


def log_params(f, header, pi, A, B):
    """
    Helper function to write parameters nicely into the log file.

    Parameters
    ----------
    f : file object
        Open text file for writing.
    header : str
        A small header/title to print before the parameters.
    pi : list[float]
        Initial distribution.
    A : list[list[float]]
        Transition matrix.
    B : list[list[float]]
        Emission matrix (rows = states, cols = [H, T]).
    """
    f.write(header + "\n")
    f.write("pi:\n")
    f.write("  " + " ".join(f"{x:.6f}" for x in pi) + "\n\n")

    f.write("A (rows = from state, cols = to state):\n")
    for i, row in enumerate(A, start=1):
        row_str = " ".join(f"{x:.6f}" for x in row)
        f.write(f"  from state {i}: {row_str}\n")
    f.write("\n")

    f.write("B (rows = states 1..3, cols = [H, T]):\n")
    for j, row in enumerate(B, start=1):
        row_str = " ".join(f"{x:.6f}" for x in row)
        f.write(f"  state {j}: {row_str}\n")
    f.write("\n")


def main():
    # ---------------- Basic settings ----------------
    NUM_EPOCHS = 5  # number of Viterbi-training epochs (change if you want)

    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", "hmm_viterbi_training_log.txt")

    # ---------------- HMM definition (Problem 2) ----------------
    # States are labeled as 1, 2, 3
    state_labels = [1, 2, 3]

    # Observations: 'H' and 'T'
    obs_symbols = ['H', 'T']
    obs_map = {'H': 0, 'T': 1}

    num_states = len(state_labels)
    num_obs = len(obs_symbols)

    # Initial distribution pi
    pi = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    # Transition matrix A (rows = from-state, cols = to-state)
    A = [
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.2, 0.1, 0.7],
    ]

    # Emission matrix B (rows = states 1..3, cols = [H, T])
    B = [
        [0.5, 0.5],    # state 1: P(H)=0.5,  P(T)=0.5
        [0.7, 0.3],    # state 2: P(H)=0.7,  P(T)=0.3
        [0.25, 0.75],  # state 3: P(H)=0.25, P(T)=0.75
    ]

    # Training observation sequence from Question (c)
    obs_seq = list("HHTTTHHTTTTTHHHH")

    # ---------------- Open log file and write header ----------------
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Viterbi-training for HMM (Problem 2(c))\n")
        f.write("=======================================\n\n")
        f.write(f"Observation sequence: {''.join(obs_seq)}\n")
        f.write(f"Number of epochs: {NUM_EPOCHS}\n\n")

        # Log initial parameters (epoch 0)
        log_params(f, header="Initial parameters (epoch 0):", pi=pi, A=A, B=B)

        # ---------------- Viterbi-training epochs ----------------
        for epoch in range(1, NUM_EPOCHS + 1):
            # E-step: Viterbi decoding with current parameters
            best_path, best_prob = viterbi(pi, A, B, obs_seq, obs_map, state_labels)

            # M-step: update parameters using decoded path
            pi_new, A_new, B_new = reestimate_parameters(
                path_labels=best_path,
                obs_seq=obs_seq,
                obs_map=obs_map,
                num_states=num_states,
                num_obs=num_obs,
                pi_old=pi,
                A_old=A,
                B_old=B,
            )

            # Overwrite parameters with updated ones
            pi, A, B = pi_new, A_new, B_new

            # Log results for this epoch
            f.write(f"Epoch {epoch}\n")
            f.write("--------\n")
            f.write(f"Best Viterbi path (states): {best_path}\n")
            f.write(f"Joint probability of best path: {best_prob:.10e}\n\n")

            log_params(
                f,
                header=f"Parameters after epoch {epoch}:",
                pi=pi,
                A=A,
                B=B,
            )

        # ---------------- Final summary ----------------
        f.write("Final parameters after all epochs:\n")
        f.write("---------------------------------\n")
        log_params(f, header="Final (same as last epoch):", pi=pi, A=A, B=B)

    # The log file now contains:
    # - initial parameters
    # - per-epoch path and probability
    # - per-epoch parameters
    # - final parameters


if __name__ == "__main__":
    main()
