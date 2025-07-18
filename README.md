
## Context

This code supports paper *Shielded Controller Units for RL with Industrial Constraints Applied to Remote Microgrids* submitted at AAAI 2026 Special Track on AI for Social Impact. It contains the code to reproduce its results by training and evaluating an RL agent on an realistic industrial environment while garanteeing industrial constraint.


## Content

`main.py` contains the code to train the agent using the stable_baselines3 RL library. You will need to be connected to WandB for it to run, or use the CLI argument `disable_wandb`.

`configs/config.yaml` contains the default configuration parameters for the agent, training process, and environment. These can be changed in `main.py` by changing them in the CLI **TODO: ADD EXAMPLE**

`env/env_microgrid.py` is the environment class simulating a microgrid, . In the environment, the shielded controller unit approach (described in the paper) has been used to garantee the respect of all operational constraints, including the generation/load balance, at every time step. The data used for demand and available wind power is real world data, normalized, accessible in `data/`. The environment can be tested/discovered with notebook `env/test_env.ipynb`.

In `agents/`, baseline heuristic policies have been implemented for performance comparison. They can be tested using `test_baselines.ipynb`. The repo also hosts the definition of RL agents. 

`deploy.py`, `evaluate.py` and `plot_trajectory.py` do **TODO: EXPLAIN WHAT THEY DO EXACTLY AND THEIR DIFFERENCES**


## Install instructions

You can install the requirements with:
`pip install -r ./requirements.txt`

**TO DO: double check**
    


## 📄 License

This repository uses a dual licensing model:

- The **source code** is licensed under the MIT License.
- The **data files** in the `data/` directory are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
