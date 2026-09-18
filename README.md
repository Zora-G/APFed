# APFed

APFed: Anti-Poisoning Attacks in Privacy-Preserving Heterogeneous Federated Learning, IEEE TIFS, 2023.

## Installation

```bash
pip install -r requirements.txt
```

## Run

MNIST example. Provide `initial_model.pt` (pretrained global model, clean reference and clusters) and `runtime_inputs.json` (data partition and training seeds).

```bash
export PYTHONHASHSEED=0
python main.py --attack sign --gpu 0 --checkpoint ./initial_model.pt --inputs ./runtime_inputs.json
```

Use `--attack none` to run without attacks.
