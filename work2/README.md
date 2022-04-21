# Work 2

First, create a virtual environment and install the necessary packages:
```bash
python -m venv venv
. venv/bin/activate

pip install --upgrade pip
pip install --editable lib
pip install matplotlib numpy scikit-learn

# this is to install pytorch on cpu
# if you have a gpu, get install instructions at `https://pytorch.org/get-started/locally/#start-locally`
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

Then, run the train and test scripts. `dataset` can be one of `MNIST` or `FashionMNIST`,
while `model` can be from `NN1` to `NN5`.
```bash
dataset=MNIST
model=NN1

python bin/train.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --model_path "data/${dataset}_${model}_weights.pt" \
    --plot_path "data/${dataset}_${model}_history.png"

python bin/test.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --model_path "data/${dataset}_${model}_weights.pt" \
    --confusion_matrix_path "data/${dataset}_${model}_confusion_matrix.png"
```
