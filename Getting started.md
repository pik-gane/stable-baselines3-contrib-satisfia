# Getting started

This document was written by me ([Clément Dumas](https://github.com/Butanium)) in order to make it easier for people to
continue this project after the end of my internship.

## Cluster access

To access the cluster with ssh, check this
guide: https://www.pik-potsdam.de/en/institute/about/it-services/hpc/user-guides/access
TLDR: `ssh -XC pikusername@cluster.pik-potsdam.de` once you set up your ssh key as described in the guide.

## Setting up your environment on the cluster

Once you are logged in, you can set up your environment by running the following commands:

```bash
module load anaconda/2021.11d
export PYTHONPATH=${PYTHONPATH}:/p/projects/ou/labs/gane/satisfia/ai-safety-gridworlds-satisfia:/p/projects/ou/labs/gane/satisfia/stable-baselines3-contrib-satisfia
source activate /p/projects/ou/labs/gane/satisfia/py310-env/
cd /p/projects/ou/labs/gane/satisfia/stable-baselines3-contrib-satisfia/
export WANDB_API_KEY=your_api_key
```

You can add these commands to your `.bashrc` file to run them automatically when you log in. To do that
run `vim ~/.bashrc` and add the previous commands to the file (you can check a vim tutorial if you don't know how to use
it).

Explanation of the commands:

- `module load anaconda/2021.11d`: loads anaconda
- `export PYTHONPATH...`: adds the python packages to the python path
- `source activate ...`: activates the python environment
- `cd ...`: changes the current directory to the project directory
- `export WANDB_API_KEY=your_api_key`: sets the wandb api key so that you can use wandb to log your experiments. This is
  optional, you can remove this line if you don't want to use wandb.
  See [wandb documentation](https://docs.wandb.ai/quickstart) for more details. But you can just use the
  command `wandb sync {directory}` to sync an experiment directory with wandb.

## Remote development

I'm not sure if you can use VSCode / PyCharm remote server on the cluster, at least I know someone who tried but ended
up using the solution I'll present you. But feel free to try it out if you want.
This solution is using PyCharm remote interpreter feature. You can check their
tutorial [here](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html#ssh). I needed a
little hack in order to make it work. You just have to select `/p/projects/ou/labs/gane/satisfia/python` as the remote
interpreter  (see [this issue](https://youtrack.jetbrains.com/issue/PY-35978/Support-conda-with-remote-interpreters) for
more details). You basically have to clone the project on your own machine, make sur that you're on the same branch as
the cluster version and follow the tutorial.

PS: PyCharm is free for students and academic research, you can get a
license [here](https://www.jetbrains.com/community/education/#students). It has more features than VSCode but it's also
more resource intensive.

## Running experiments

To run experiments on the cluster you can use slurm. This is a job scheduler that will run your experiments on the
cluster. I've already coded an interface but you can check
the [PIK guide](https://www.pik-potsdam.de/en/institute/about/it-services/hpc/user-guides/slurm) and
the [slurm documentation](https://slurm.schedmd.com/documentation.html) if you want to know more about it, but maybe you
won't need it as I've already coded an interface.

To run an experiment you basically use the `submit_job_array` function from `./experiments/cluster/slurm.py`. For an
example of how to use it check `./experiments/cluster/ardqn_experiment.py` which submit a job array running
the `./experiments/cluster/ardqn_worker.py` script. Check the docstring of `submit_job_array` for more details.

## Experiment logs

For now, scalar logs are collected with [`tensorboard`](https://www.tensorflow.org/tensorboard) and stored
in `./experiments/logs/{experiment_name}`. You can run `tensorboard --logdir ./experiments/logs/{experiment_name}` to
visualize them. After logging in `wandb` / `tensorboard` (see there doc for more info), you can upload them using
either `wandb sync ./experiments/logs/{experiment_name}`
or `tensorboard dev upload --logdir ./experiments/logs/{experiment_name}`. The first one will upload the logs to wandb
and the second one to tensorboard.dev. I've used wandb until it stopped working for me, so I switched to
tensorboard.dev. Also, it seems like we can't run the tensorboard visualization on the cluster, so you'll have to
download the logs on your computer to visualize them.

## Code structure

TLDR: Our ar_dqn algorithm is in `sb3_contrib/ar_dqn` which inherit from `sb3_contrib/common/satisficing`

The code is structured as follow:

```bash
├── experiments 
│   ├── # Some stuff to test the code (not important)
│   ├── cluster  # Cluster related experiments
│   │   ├── ardqn_experiment.py  # ARDQN experiment: submit a job array to run ardqn_worker.py
│   │   ├── ardqn_worker.py  # ARDQN worker: run an instance of ardqn
│   │   ├── ardqn_post_experiment.py  # ARDQN post experiment: analyze the results of ardqn_experiment.py
│   │   ├── ...  # Other experiments
│   │   ├── array_template.sh  # Template file filled by submit_job_array
│   │   └── slurm.py  # Interface to submit jobs to slurm
│   ├── custom_envs.py  # Some custom envs
│   ├── public_good_envs  # Jobst envs
│   │   ├── public_good.py  # A public good game with linear benefits and quadratic individual costs shared proportionally
│   │   └── iterated_pd.py  # Iterated prisoner's dilemma
│   └── utils.py  # Some ways to open tensorboard from code and a callback to Make DQN tensorboard similar to AR-DQN
├── sb3_contrib  # Library directory
│   ├── ar_dqn  # Modified version of DQN derived from common.satisficing
│   │   ├── ar_dqn.py
│   │   └── policies.py
│   ├── ar_q_learning  # Modified version of Q learning
│   │   ├── ar_q_learning.py
│   │   └── policies.py
│   ├── q_learning  # My Q learning 
│   │   ├── policies.py
│   │   └── q_learning.py
│   ├── common
│   │   ├── satisficing  # Unified interface for both Q learning and DQN
│   │   │   ├── type_aliases.py
│   │   │   ├── algorithms.py  # Implements the update function / log function and predict function
│   │   │   ├── buffers.py  # Add the lambda storage
│   │   │   ├── evaluation.py  # Evaluate the satisficing policy by performing aspiration rescaling between each steps. Also contains a plot_ar function to plot some results
│   │   │   ├── policies.py  #  Implements rescale aspiration
│   │   │   └── utils.py  # Contains the x:y:z and x/y/z operators and a decorator for gathering Q values with actions.
│   │   ├── wrappers
│   │   │   ├── ...
│   │   │   └── time_feature.py  # Might be useful to add time to the agent observation. I also have my own implementation but maybe this one is better
│   │   └── ...  # Stuff related to other algorithms implemented in the library
│   ├── ...  # Other algorithms already implemented in the library
├── scripts
│   └── run_tests.sh  # Useful to check your code
├── setup.py
└── tests  # A folder containing tests that you can run to check if you didn't break anything. Disclaimer: I mostly just adapted existing test to ARDQN. 
    │      # I also disabled the tests of the other algorithms to speed up the test. As I'm writing those lines there are still some tests that don't pass :)
    └── ...
```

## AR-DQN parameters

Throughout this internship, I experimented with numerous hyperparameters in an effort to stabilize the learning process.
Some of these might not significantly affect the outcome, or a single value may outperform the others and thus become
the default.
If you identify such cases, please feel free to eliminate these hyperparameters. Streamlining our parameters for optimal
performance is always beneficial.

todo: Explain the parameters (if it's still a todo while you're reading this, you can check the docstring of
the `AR_DQN` class in `sb3_contrib/ar_dqn/ar_dqn.py`)
Hyperparameters specific to AR-DQN:

- `rho`
- `mu`
- `shared_network`
- `q_min_max_target`
- `use_delta_net`

Hyperparameters shared with DQN:

- `exploration_fraction`
- `exploration_initial_eps`
- `exploration_final_eps`
- `learning_rate`
- `learning_starts`
- `target_network_update_freq`
- `gamma`

## Suggested tasks / ideas

- [ ] Add a training procedure for LRAR-DQN. For now it's using pretrained LRA-DQN Q network, but it would be great to
  improve those Q networks on the actual LRAR-DQN trajectories. I think you'd have to make inherit from the DQN class.

## Tasks that I didn't have time to do but I though might be useful (feel free to do them if you want)

- [ ] Test the iterated prisoner dilemna envs
- [ ] Implement and test the Jobst snooker env
- [ ] Make the plot_ar function in `sb3_contrib/common/satisficing/evaluation.py` more generic, so that you can
  partition the models based on arbitrary parameters (for now it's only "performance as a function of
  mu/rho/aspiration"). This could be done by using pandas dataframes
  Also it would be useful to make it interactive using dash to e.g filter curves with regex patterns. I did a toy
  example
  of such regex filter app [here](https://pastebin.com/Nkk3V2PM).
- [ ] Add some tests for the AR-Q-learning algorithm and LRA
- [ ] Optimize the code of the algorithms to make it faster. I mainly focus on making the code readable but I think
  there is still some room for optimization.
- [ ] Fix the `test_save_load` for AR-DQN when `shared_network` is either `all` or `min_max`. I think the error
  comes from the fact that parameters are included multiple times, but I'm not sure if it's a problem in practice.
- [ ] Try some hyperparameter optimization on the AR-DQN parameters and try to find the parameters that really matter

## Notes

- Before launching a cluster experiment, try it on a small number of workers. At the beginning I made some 500 workers
  experiments that ran for 3 minutes because there was a syntax error in my code. The problem is that, after that,
  the cluster was way longer to give me access to resources.
- If you want to merge this repo in sb3-contrib, make a clean branch and :
    - Follow the CONTRIBUTING.md guidelines
    - remove the `experiments` folder and this `Getting started.md` file
    - update the `README.md` and `setup.py`
    - make sure that the AR-DQN tests pass
    - revert my changes that disabled the tests of the other algorithms
    - maybe separate the AR-DQN algorithm and the (AR) Q-learning algorithms in two different PR

## Questions ?

Feel free to reach me out, I'm happy to help with this project. Also feel free to ask me to review your first PRs, so
that I can give you some feedbacks on your code :)