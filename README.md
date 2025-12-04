# Deep RL for Quantum Circuit Optimization

This repository implements a Deep Reinforcement Learning (RL) framework to optimize quantum circuits for state and operator synthesis tasks. We train a Deep Q-Network (DQN) agent aiming to automate the discovery of the most efficient (shortest gate-depth) quantum circuit for either a state transformation or unitary operator. This problem is relevant in the NISQ era, where minimizing circuit depth is directly linked to reducing error from noise and decoherence, and a higher execution fidelity.

## Approach

This project frames the quantum circuit design problem as a Markov Decision Process (MDP) solved using a Deep Q-Network (DQN) agent. The primary technical challenge is managing the exponential complexity of the quantum state space. We mitigate this by representing the environment state as a non-exponentially scaling feature vector, the expectation values of $3N$ single-qubit Pauli operators ($X$, $Y$, $Z$ for each of the $N$ qubits). This compact representation allows the DQN's Q-network to learn the appropriate policy efficiently. Training stability is ensured through the use of Experience Replay and Double DQN techniques, including a separate Target Network for calculating stable loss targets.

The training objective transitions from State Synthesis (matching a target state vector, in our case $|GHZ\rangle$) to the more challenging Unitary Synthesis (matching a target operator, in our case a Quantum Fourier Transform $U_{QFT}$). This requires switching the reward metric from State Fidelity to Unitary Fidelity. Crucially, we employ Reward Shaping by giving a proportional reward for any step that increases fidelity, along with a penalty for circuit depth. This dense feedback mechanism is for guiding the agent to discover minimal-depth circuits that achieve near-perfect synthesis of complex operations.

## Usage

To use this repository first clone and install the required dependencies ([uv](https://docs.astral.sh/uv/) recommended):

```bash
git clone https://github.com/SepehrAkbari/qgate-dqn.git
cd qgate-dqn
uv venv .venv
uv activate .venv
uv pip install -r requirements.txt
```

This project is split into two sources, one to train an agent for state synthesis and another for operator synthesis. Both of the sources have to be ran using `python -m src.<SOURCE_NAME>`, where `<SOURCE_NAME>` is either `GHZ` for state synthesis or `QFT` for operator synthesis. If you wish to use the pre-trained models, available in [model](./model/):

```bash
python -m src.<SOURCE_NAME>.infer
```

You can also train the models from scratch. In that case, manually remove any existing model files (`*.pth`) from the model directory before running the training script:

```bash
python -m src.<SOURCE_NAME>.train
python -m src.<SOURCE_NAME>.infer
```

Additionally, both sections have a corresponding Jupyter notebook in the [notebook](./notebook/) directory that walks through the implementation details and evaluation of the trained agents. Note that the operator synthesis is dependent on the state synthesis implementation, so it is recommended to run the state synthesis notebook first.

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](./LICENSE).
