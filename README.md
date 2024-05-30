# explore-sae
Exploring the use of sparse auto-encoders (SAEs), eventually in parsing formal languages.

# Files and Directories
## Training and Saving Models for Modular Addition

* `hparams.py`: Specify data split, $p$, model hyperparameters, and training hyperparameters.
* `data.py`: Generate training and testing data for addition.
* `model.py`: Define `torch_model` (off-the-shelf transformer) and `custom_model` (Nanda's custom transformer; doesn't use LayerNorm).
* `utils.py`: Some helpers, including plotting code.
* `train.py`: Training and saving loop.

The models (checkpoints every 100 epochs) are saved in the `save/` folder:

* `save/grok_1716823448/`: Running on the off-the-shelf transformer.
* `save/grok_1716836929/`: Running on the custom transformer.

## Training and Saving Autoencoders

* `autoencoder.py`: The definition of the autoencoder model, its data and its training loop.
* `sweep.py`: A script to sweep over L1 regularization coefficients in the range $10^{-3} \leq \alpha \leq 5 \cdot 10^2$, and compare the reconstruction and regularization losses they achieve [see below].


The data used to train the autoencoder is saved in the format `activations/{run_name}/{ckpt}_{layer_name}.pth`.  
`run_name` is the directory name of the model under `save/`; `ckpt` is the epoch number or `final`; and `layer_name` is the layer whose output is stored.

The trained models are saved in the same format.

# Uncommitted Info
## Storing Activations
Activations are stored through a notebook `analysis.py`.

## Regularization Coefficient for SAE
We use Method 2 of [Taking features out of superposition with sparse autoencoders](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition), where the reconstruction and regularization losses are both plotted and we find the $\alpha$ at which they both plateau.