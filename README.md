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

The data used to train the autoencoder is saved in the format `activations/{run_name}/{ckpt}_{layer_name}.pth`.  
`run_name` is the directory name of the model under `save/`; `ckpt` is the epoch number or `final`; and `layer_name` is the layer whose output is stored.

The trained models are saved in the same format.

# Uncommitted Info
## Storing Activations
Activations are stored through a notebook `analysis.py`.

## Regularization Coefficient for SAE
We do a sweep on $\alpha \in \{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 10^2\}$ for each latent size. These weights give the following MSE losses (respectively) after 10,000 epochs.

| Latent size | $10^{-6}$ | $10^{-5}$ | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ | 1 | 10 | 100 |
| :---:       | :---:     | :---:     | :---:     | :---:     | :---:     | :---:     | :---: | :---: | :---: |
| 128         | $1.14 \cdot 10^{-5}$ | $1.49 \cdot 10^{-5}$ | $1.96 \cdot 10^{-5}$ | $3.99 \cdot 10^{-5}$ | $3.7 \cdot 10^{-4}$ | $1.08 \cdot 10^{-3}$ | $1.08 \cdot 10^{-3}$ | $1.51 \cdot 10^{-3}$ | $3.76 \cdot 10^{-2}$ |