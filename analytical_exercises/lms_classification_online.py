import os

# -------------------------------------------------------------------
# Online LMS (Widrow–Hoff) in 1D, 3 epochs
# -------------------------------------------------------------------

# Training data: 1D inputs and targets (A -> +1, B -> -1)
xs = [1, 2, 3, 4, 5, 6]            # inputs
ts = [+1, -1, +1, -1, +1, -1]      # targets

# Hyperparameters
eta = 0.01                         # learning rate
epochs = 3                         # number of passes over the dataset

# Parameter initialization
w = 0.1                            # initial weight
b = 0.1                            # initial bias

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Output file path
output_path = os.path.join("results", "lms_online_log.txt")

# Open output file (overwrite each run)
with open(output_path, "w") as f:
    # Global header
    f.write("Online LMS training log (per iteration and per epoch)\n")
    f.write("=" * 90 + "\n\n")

    # Main training loop over epochs
    for epoch in range(1, epochs + 1):
        # ---------------------------------------------------------
        # 1) Compute INITIAL cost before any updates in this epoch
        #    J_initial = (1/2) * sum_n (t_n - y_n)^2 with current w, b
        # ---------------------------------------------------------
        J_initial = 0.0
        for x, t in zip(xs, ts):
            y = w * x + b          # current prediction
            e = t - y              # current error
            J_initial += 0.5 * (e ** 2)

        # ---------------------------------------------------------
        # Print epoch header and per-iteration table header
        # ---------------------------------------------------------
        f.write(f"Epoch {epoch}\n")
        f.write("-" * 90 + "\n")
        # Column headers for per-iteration table
        f.write(
            " i  |  x  |  t  |   w_before  |   b_before  |"
            "       y     |       e     |   J_sample  | dJ_dw_samp | dJ_db_samp |   w_after   |   b_after   \n"
        )
        f.write("-" * 90 + "\n")

        # ---------------------------------
        # 2) ONLINE UPDATES for this epoch
        #    One LMS update per sample
        # ---------------------------------
        for i, (x, t) in enumerate(zip(xs, ts), start=1):
            # Parameters before the update
            w_before = w
            b_before = b

            # Forward pass with current parameters
            y = w_before * x + b_before
            e = t - y

            # Instantaneous loss for this sample
            J_sample = 0.5 * (e ** 2)

            # Instantaneous gradients (for the sample loss J_n)
            # dJ/dw = -(t - y) * x = -e * x
            # dJ/db = -(t - y)     = -e
            dJ_dw_sample = -e * x
            dJ_db_sample = -e

            # Online LMS update (Widrow–Hoff rule)
            w_after = w_before + eta * e * x
            b_after = b_before + eta * e

            # Commit the updated parameters
            w = w_after
            b = b_after

            # Log this iteration as an aligned table row
            f.write(
                f"{i:3d} |"
                f" {x:3d} |"
                f" {t:3d} |"
                f" {w_before:11.6f} |"
                f" {b_before:11.6f} |"
                f" {y:11.6f} |"
                f" {e:11.6f} |"
                f" {J_sample:11.6f} |"
                f" {dJ_dw_sample:10.6f} |"
                f" {dJ_db_sample:10.6f} |"
                f" {w_after:11.6f} |"
                f" {b_after:11.6f}\n"
            )

        # -------------------------------------------------
        # 3) Compute FINAL cost and gradients after updates
        #    J_final    = (1/2) * sum_n (t_n - y_n)^2
        #    dJ_dw_last = - sum_n (t_n - y_n) * x_n
        #    dJ_db_last = - sum_n (t_n - y_n)
        # -------------------------------------------------
        J_final = 0.0
        dJ_dw_final = 0.0
        dJ_db_final = 0.0

        for x, t in zip(xs, ts):
            y = w * x + b          # prediction with UPDATED parameters
            e = t - y

            # Accumulate final cost
            J_final += 0.5 * (e ** 2)

            # Accumulate final batch gradients at end of epoch
            dJ_dw_final += -e * x
            dJ_db_final += -e

        # ------------------------------
        # 4) Epoch-level summary table
        # ------------------------------
        f.write("-" * 90 + "\n")
        f.write("Epoch summary:\n")
        f.write(
            "   w_final   |   b_final   |  J_initial  |   J_final   | dJ_dw_final | dJ_db_final\n"
        )
        f.write(
            f" {w:11.6f} | {b:11.6f} |"
            f" {J_initial:11.6f} | {J_final:11.6f} |"
            f" {dJ_dw_final:11.6f} | {dJ_db_final:11.6f}\n"
        )
        f.write("\n\n")
