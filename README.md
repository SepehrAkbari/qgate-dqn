# Deep RL for Quantum Circuit Optimization

This repository implements a Deep Reinforcement Learning (RL) framework to optimize quantum circuits for state and operator synthesis tasks. We train a Deep Q-Network (DQN) agent aiming to automate the discovery of the most efficient (shortest gate-depth) quantum circuit for either a state transformation or unitary operator. This problem is relevant in the NISQ era, where minimizing circuit depth is directly linked to reducing error from noise and decoherence, and a higher execution fidelity.

## Approach



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

Additionally, both sections have a corresponding Jupyter notebook in the [notebook](./notebook/) directory that walks through the implementation details and evaluation of the trained agents.

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](./LICENSE).
