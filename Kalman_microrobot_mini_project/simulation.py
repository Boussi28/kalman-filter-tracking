import numpy as np
import matplotlib.pyplot as plt


#reproducibility 
np.random.seed(42)

#number of frames 
n_frames = 100

#simulating a scenario where detection is lost for few frames
dropout_start = 40
dropout_end = 55

#start position (position and velocity)
x, y = 50.0, 50.0
vx, vy = 2.0, 1.0

#storing true path
true_positions = []

for frame in range (n_frames):
    vx += np.random.normal(0, 0.1)
    vy += np.random.normal(0, 0.1)

    x += vx
    y += vy

    #recording the position at this frame
    true_positions.append((x, y))

#converting the list to an array
true_positions = np.array(true_positions)

#adding noise to true positions to simulate measurements
measurement_noise = 5.0

measurements = []

for pos in true_positions:
    mx = pos[0] + np.random.normal(0, measurement_noise)
    my = pos[1] + np.random.normal(0, measurement_noise)
    measurements.append((mx, my))

measurements = np.array(measurements)

#Kalman filter
# --- Step 3: Kalman filter ---

dt = 1.0  # time between frames (1 frame)

# Motion model matrix
# state = [x, y, vx, vy]
# new x  = x + vx*dt ; new y = y + vy*dt ; velocities unchanged
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1,  0],
    [0, 0, 0,  1]
])

# Measurement matrix H: we only measure position (x, y), not velocity
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# Process noise covariance Q: how much the motion can deviate from the model
# small values = we trust the constant-velocity model
Q = np.eye(4) * 0.01

# Measurement noise covariance R: how noisy the detector is
# we set it near the real measurement noise (5.0), so variance ~ 5^2
R = np.eye(2) * 25.0

# Initial state estimate: start at the first measurement, zero velocity
state = np.array([measurements[0, 0], measurements[0, 1], 0, 0], dtype=float)

# Initial estimate uncertainty P: large, because we're unsure at the start
P = np.eye(4) * 500.0

# We'll store the filter's estimated positions here
estimates = []

for frame in range(n_frames):
    # --- PREDICT (always happens) ---
    state = F @ state
    P = F @ P @ F.T + Q

    # --- UPDATE (only if detection succeeded this frame) ---
    if dropout_start <= frame < dropout_end:
        # Detection failed: no measurement, skip the update.
        # The filter coasts on prediction alone.
        pass
    else:
        z = measurements[frame]
        z = np.array([z[0], z[1]])
        y_residual = z - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ y_residual
        P = (np.eye(4) - K @ H) @ P

    # Record the estimated position every frame, detected or not
    estimates.append((state[0], state[1]))

estimates = np.array(estimates)

# Quantifying the tracking accuracy against the known ground truth
errors = np.sqrt(np.sum((estimates - true_positions)**2, axis=1))
rmse = np.sqrt(np.mean(errors**2))
print(f"RMSE of Kalman estimate vs true path: {rmse:.2f} pixels")

# For comparison, how noisy the raw measurements were
meas_errors = np.sqrt(np.sum((measurements - true_positions)**2, axis=1))
meas_rmse = np.sqrt(np.mean(meas_errors**2))
print(f"RMSE of raw measurements vs true path: {meas_rmse:.2f} pixels")

#plotting the true path
plt.figure(figsize=(8, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True path')
detected_frames = [f for f in range(n_frames) if not (dropout_start <= f < dropout_end)]
plt.scatter(measurements[detected_frames, 0], measurements[detected_frames, 1],
            c='red', s=15, label='Noisy measurements')
plt.plot(estimates[:, 0], estimates[:, 1], 'b-', linewidth=2, label='Kalman estimate')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Kalman filter tracking')
plt.legend()
plt.axis('equal')
plt.savefig('kalman_tracking_dropout.png', dpi=150, bbox_inches='tight')
plt.show()
