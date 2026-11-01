# Reproducibility and interpretation

## Default experiment

Both self-contained notebooks use seed 7 and gamma 1. Each MC, SARSA, Q-learning and REINFORCE run uses 20,000 episodes. PPO uses 200 batches of 100 episodes with four full-batch optimisation epochs per batch. These are selected teaching settings, not the book v21 experiment settings or a tuned benchmark.

MC uses an incremental sample mean per visited state/action pair. SARSA and Q-learning use step size 0.15. Their behaviour is epsilon-greedy, with epsilon decreasing linearly from nearly one to a floor of 0.05 during the first 80% of training. REINFORCE uses actor step size 0.08. PPO uses actor and critic steps of 1.0, clipping 0.2, a tabular softmax actor and a tabular critic. PPO advantages are complete observed return minus frozen old state value. Old probabilities, returns and advantages remain fixed for all four epochs; gradients are summed within games and averaged over the number of games.

In both games a state cannot repeat within an episode. This makes first-visit and every-visit MC identical and lets REINFORCE update distinct table rows without altering another record's policy probability in the same episode.

Q-table snapshots are independent copies at 0, 1,000 and final training episodes. No trained arrays are secretly loaded. Tables, rules and parameters are all present in the notebooks. Displayed numbers are rounded; action selection uses unrounded numbers with the first exact maximum breaking ties.

## Evaluation protocol

Tests freeze the learned parameters and use a separate random stream (seed 1007). There are 500 fresh games per reported row. Nim tests start with 9 matches; Tic-Tac-Toe tests start with an empty board and learner X. Actor results distinguish largest-probability deployment from sampling. No evaluation result is used to choose the seed or to update the tables.

The perfect-play solver is defined after the training calls. A separate exact worst-case calculation explores every opponent reply with the learned greedy player frozen. It can reveal losses missed by sampled test games. The exact reference is a verification tool, not a learner input. Nim also compares selected actions with an exact optimum against its declared training opponent; that is a different optimum from adversarial game play.

Training curves include exploration and different initial-state distributions from some tests. Equal episode counts do not imply equal computational work. Single-seed examples and 500-game proportions do not establish a universal ranking, precise population rates or convergence. Constant learning rates and finite exploration schedules are deliberate simplifications.

## Re-run

In Colab, start a fresh runtime and use **Runtime → Run all**. The notebooks have no repository imports and no network/data-loading cells. No interactive prompt blocks a full run: human play is opt-in.

Locally, install `requirements.txt`, open JupyterLab and restart/run all cells. Training and scientific packages can change floating-point details between versions. Broad dependency ranges support ordinary current notebook environments; they do not promise bitwise-identical results on every platform.

Local validation environment: Python 3.9.13, NumPy 1.21.6, pandas 1.4.4, Matplotlib 3.5.2, nbformat 5.5.0, nbclient 0.5.13 and IPython 7.31.1. Use these versions only if reconstructing this older validation environment; a normal Colab runtime need not be downgraded.

Run the independent checks from the repository root:

```text
python tests/check_notebooks.py
```

The checks verify environment boundaries, legal-action masks, numerical updates, PPO gradients and game outcomes. They use reduced training runs for automated checks. Committed notebook outputs come from full default training runs. See `validation.json` for the actual completion status and measured results.

## Scope

No Gymnasium, PyTorch, TensorFlow or RL framework is required. PPO is a small, explicit tabular PPO-Clip demonstration, not a neural-network agent. There is no symmetry reduction or self-play. The Tic-Tac-Toe learner plays X only. Changing gamma, perspective, reward timing or opponent policy requires reviewing the equations and evaluation assumptions together.
