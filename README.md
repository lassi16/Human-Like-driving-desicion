Human-like Driving Decision System (Prototype)
=============================================

This project implements a simplified, end-to-end prototype of a
"Human-like Driving Decision System" inspired by the paper
"Humanlike Driving: Empirical Decision-Making System for Autonomous Vehicles".

Modules
-------
- data_generator.py
  * Generates random grid-based traffic scenes (10x5).
  * Each cell contains a vehicle type and speed.
  * Provides heuristic labeling logic to mimic human-like decisions
    (speed + steering).

- model.py
  * Defines a fully-connected Decision-Making Network (DMN) in PyTorch.
  * Shared trunk with layers: 1028 -> 1028 -> 512 units.
  * Two output branches: normalized speed (regression) and steering
    logits (3-way classification for left/straight/right).

- safety_layer.py
  * Implements a simplified repulsive potential field (RPF) style
    safety layer.
  * Adjusts speed and steering preferences based on nearby obstacles.
  * Combines NN output with safety suggestions.

- utils.py
  * Helper utilities: seeding, steering string conversion, collision
    estimation, grid visualization, and scene flattening.

- train.py
  * End-to-end training script using synthetic data.
  * Uses Adam optimizer, MSE loss for speed, CrossEntropy loss for
    steering, and combined total loss.
  * Saves the trained PyTorch model to dmn_model.pth.

- simulation.py
  * Loads a trained model.
  * Runs new random scenarios and visualizes the grid, NN decision,
    safety-only decision, and final combined decision.
  * Estimates collision rate with and without the safety layer.

How to Run
----------
1. Install dependencies (in a virtualenv or conda env):

   pip install torch matplotlib numpy

2. Train the decision-making network:

   python train.py

   This will generate synthetic scenarios, train the DMN for several
   epochs, and save the model to dmn_model.pth.

3. Run the simulation and visualization:

   python simulation.py

   This will:
   - Sample new random scenes.
   - Show predicted NN speed + steering.
   - Apply the safety layer and show final corrected decisions.
   - Visualize the grid with ego and surrounding vehicles.
   - Report approximate collision rates with and without the safety layer.

Notes
-----
- The system operates on an abstract grid representation and does not
  use any raw sensor data, LiDAR, or CNN perception.
- The focus is on the decision-making stage and demonstrating how a
  learned policy and a safety layer can be combined to yield more
  human-like and safer behavior.
- All logic is modular and can be extended to richer scenarios or
  integrated with a simulator such as CARLA.

Detailed System Description
---------------------------

At a high level, the project builds a supervised learning pipeline
around synthetic driving scenes:

1. Scenario generation and pseudo "human" labels
   - The road is abstracted as a 10x5 grid. Each cell stores a
     discrete vehicle type (empty, car, truck, ego) and a speed.
   - Random scenes are created by placing the ego car on the bottom
     row and populating other cells with vehicles at random speeds.
   - A hand-crafted heuristic (Data Labeling Logic) inspects these
     scenes and outputs what a "reasonable human" might do:
       * Target speed (slow down if there is a slow/close obstacle
         ahead, speed up a bit if the lane is free).
       * Steering decision (left/right/straight) depending on which
         lanes ahead look safer or more open.
   - These heuristic outputs act as ground-truth labels for training
     the neural network, so the network learns to imitate this
     human-like rule set rather than rigid traffic rules.

2. Decision-Making Network (DMN)
   - Each grid is flattened into a feature vector that concatenates
     normalized types and speeds. This is the sole input; no images or
     sensor data are used.
   - The DMN is a fully connected neural network in PyTorch with a
     shared trunk (1028 -> 1028 -> 512) followed by two heads:
       * Speed head: outputs a scalar normalized speed in [0,1], later
         scaled to 0–100 km/h.
       * Steering head: outputs 3 logits corresponding to
         left/straight/right.
   - Training uses a combined loss:
       * MSE on normalized speed (regression objective).
       * Cross entropy on steering logits (classification objective).
   - Optimizer: Adam, with typical settings and mini-batches of size
     32 for 20–50 epochs (configurable in train.py).

3. Safety layer (simplified Repulsive Potential Field)
   - After the DMN predicts speed and steering, a safety module
     examines the grid again, focusing on nearby vehicles around the
     ego car.
   - For each vehicle, a distance-based "repulsive" influence is
     computed:
       * Closer obstacles force a stronger reduction in a proposed
         safe speed.
       * Obstacles in the same or adjacent lanes reduce the
         preference for straight/left/right steering into them.
   - The neural network’s output is then blended with this safety
     output:
       * final_speed = w1 * NN_speed + w2 * safety_speed
       * final_steering_logits = w1 * NN_logits + w2 * safety_logits
     where w1 and w2 are tunable weights. This reflects the idea of a
     human-like policy corrected by a conservative safety layer.

4. Evaluation and metrics
   - During training, steering accuracy and regression loss are
     tracked on a held-out validation set.
   - In simulation, new random scenes are generated that the model has
     never seen before. For each scene, the code reports:
       * NN-only decision (speed + steering) and whether that decision
         would likely collide with an obstacle in the chosen lane
         within a short look-ahead distance.
       * Safety-only speed suggestion.
       * Final combined decision and its collision risk.
   - Over many random scenarios, an approximate collision rate is
     computed for:
       * NN-only decisions.
       * NN decisions corrected by the safety layer.
   - A reduction in this collision rate indicates that the safety
     layer is successfully making the learned policy more conservative
     and human-like.

5. Technologies and libraries used
   - Python 3.x
   - PyTorch: neural network definition, optimization, and training.
   - NumPy: numeric operations and synthetic data generation.
   - Matplotlib: 2D visualization of the grid, highlighting ego and
     surrounding vehicles, with annotations of predicted and safe
     decisions.

Together, these components create an end-to-end prototype that
captures the core idea from human-like driving research: using a
learned decision-making policy that imitates human behavior, while a
separate safety layer enforces simple physics/spacing constraints to
avoid clearly unsafe maneuvers.

How the Neural Network Works (Intuition)
----------------------------------------

- The grid in front of the ego car is turned into a long 1D vector.
  For each cell we put two numbers: one describing the vehicle type
  (empty, car, truck, ego) and one describing its speed. These are
  normalized to stay in a small numeric range.
- This vector is fed into a stack of fully-connected layers
  (DecisionMakingNetwork in model.py). You can think of this as a
  function that learns patterns such as "if there is a slow car right
  in front of me, slow down" or "if the left lane is free, consider
  steering left".
- The shared trunk of layers (1028 → 1028 → 512 units with ReLU) first
  extracts a compressed representation of the whole traffic scene.
  From this internal representation two small heads branch out:
  - Speed head: outputs one number between 0 and 1 (after sigmoid),
    which is then scaled to 0–100 km/h.
  - Steering head: outputs three scores (logits) for
    left/straight/right. The highest score is taken as the steering
    decision.
- During training (train_model in train.py), the network repeatedly
  sees synthetic scenes together with the heuristic "human-like"
  labels. The optimizer (Adam) adjusts the weights to reduce:
  - the difference between predicted and target speed (MSE loss), and
  - the classification error on steering (cross-entropy loss).
- After enough iterations, the network approximates the heuristic
  rule set: for a new, unseen grid, it can directly output a
  reasonable speed and a steering choice in one forward pass.

How the Decision-Making + Safety Module Works Together
------------------------------------------------------

- At runtime (e.g., in simulation.py), the pipeline for each new
  scene is:
  1) Flatten grid → feed into DMN → get nn_speed (normalized) and
     nn_steering_logits.
  2) Convert nn_speed to km/h and interpret nn_steering_logits as the
     NN's raw preferences for left/straight/right.
  3) Pass the same grid to the safety layer (safety_layer.py), which
     re-checks nearby vehicles and computes:
       - a "safe_speed" that brakes more if obstacles are close,
       - safety_steering_logits that heavily penalize steering into
         blocked or risky lanes.
  4) Combine both using weighted averaging:
       final_speed = w_nn * nn_speed_kmh + w_safe * safe_speed
       final_logits = w_nn * nn_steering_logits + w_safe * safety_logits
     By default, w_nn is a bit larger than w_safe, so we mostly follow
     the learned policy but still respect safety corrections.
  5) The action sent to the ego car is:
       - final_speed (clipped to [0, MAX_SPEED]) and
       - final_steering = argmax(final_logits).
- In addition to soft blending, a "hard" safety check can mark a lane
  as forbidden if there is a vehicle too close ahead in that lane. In
  that case, the corresponding steering logit is set to a very large
  negative value so it will almost never be chosen.
- This design mirrors how a human driver behaves: first form an
  intention based on experience (what the NN has learned), then adjust
  it at the last moment if something clearly unsafe is detected in the
  local surroundings.
