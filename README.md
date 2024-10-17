## Online Episodic Convex Reinforcement Learning with Adversarial Losses

Code to reproduce the results of the paper "Online Episodic Convex Reinforcement Learning with Adversarial Losses".

To run an experiment save all files to a folder, install the requirements in requirements.txt and execute

`python main.py --reward_type 'multi_objectives' --n_iterations 50 --samples 1`

Variables:
- `reward_type` can be 'multi_objectives' to run the multi-objective problem or 'constrained' to run the constrained problem;
- `n_iterations` indicates the number of iterations;
- `samples` indicates the number of repetitions per experiment.

The code runs the experiments for Greedy MD-CURL and Bonus O-MD-CURL. It creates a `results` folder containing state-action distribution images, log-loss and regret plots.

The `optimal` folder contains the values of the loss function for an approximate optimal policy to be used when calculating regret. These values are calculated using `MD-CURL`, an off-line method for CURL (using the true probability transition) from [Moreno et al. (2024)], over 2000 iterations.
