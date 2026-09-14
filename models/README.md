# Evaluation model files

Large checkpoints are not committed to this repository.

## AlignScore

Place the checkpoint at:

```text
models/AlignScore-base.ckpt
```

Install AlignScore from its upstream repository and install the required spaCy English model:

```bash
git clone https://github.com/yuh-zha/AlignScore.git
cd AlignScore
pip install .
python -m spacy download en_core_web_sm
```

## BLEURT

Place the BLEURT-20 checkpoint directory at:

```text
models/BLEURT-20/
```

Install BLEURT from the upstream Google Research repository. Its README also provides the BLEURT-20 checkpoint download instructions.

## FactSpotter

FactSpotter is loaded through Hugging Face Transformers using:

```text
Inria-CEDAR/FactSpotter-DeBERTaV3-Base
```

unless another model name is supplied on the command line.
