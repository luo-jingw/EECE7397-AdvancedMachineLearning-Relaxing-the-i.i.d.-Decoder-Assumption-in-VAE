set -e

# train
echo "train"

echo "mnist baseline (plain VAE)"
python3 train.py --dataset mnist --save-path ./mnist_baseline.pt
echo "cifar10 baseline (plain VAE)"
python3 train.py --dataset cifar10 --save-path ./cifar10_baseline.pt

echo "mnist globalK"
python3 train.py --dataset mnist --save-path ./mnist_globalK.pt --global-k --global-k-warmup-epochs 10
echo "mnist globalK sparse"
python3 train.py --dataset mnist --save-path ./mnist_globalK_sparse.pt --global-k --cov-sparse-l1-d 4 --global-k-warmup-epochs 10

echo "mnist localK"
# "full" (no sparsity) via L1 ball covering full 28x28 grid: max L1 = (28-1)+(28-1)=54
python3 train.py --dataset mnist --save-path ./mnist_localK.pt --global-k --cov-sparse-l1-d 54 --global-k-warmup-epochs 10 --local-k
echo "mnist localK sparse"
python3 train.py --dataset mnist --save-path ./mnist_localK_sparse.pt --global-k --cov-sparse-l1-d 4 --global-k-warmup-epochs 10 --local-k

echo "cifar10 globalK"
python3 train.py --dataset cifar10 --save-path ./cifar10_globalK.pt --global-k --global-k-cifar-independent --global-k-warmup-epochs 10
echo "cifar10 globalK sparse"
python3 train.py --dataset cifar10 --save-path ./cifar10_globalK_sparse.pt --global-k --global-k-cifar-independent --cov-sparse-l1-d 4 --global-k-warmup-epochs 10

echo "cifar10 localK"
# "full" (no sparsity) via L1 ball covering full 32x32 grid: max L1 = (32-1)+(32-1)=62
python3 train.py --dataset cifar10 --save-path ./cifar10_localK.pt --global-k --global-k-cifar-independent --cov-sparse-l1-d 62 --global-k-warmup-epochs 10 --local-k
echo "cifar10 localK sparse"
python3 train.py --dataset cifar10 --save-path ./cifar10_localK_sparse.pt --global-k --global-k-cifar-independent --cov-sparse-l1-d 4 --global-k-warmup-epochs 10 --local-k

# vis
python3 cov_experiment.py --ckpt ./mnist_baseline.pt
python3 cov_experiment.py --ckpt ./cifar10_baseline.pt

python3 cov_experiment.py --ckpt ./mnist_globalK.pt
python3 cov_experiment.py --ckpt ./mnist_globalK_sparse.pt
python3 cov_experiment.py --ckpt ./mnist_localK.pt
python3 cov_experiment.py --ckpt ./mnist_localK_sparse.pt

python3 cov_experiment.py --ckpt ./cifar10_globalK.pt
python3 cov_experiment.py --ckpt ./cifar10_globalK_sparse.pt
python3 cov_experiment.py --ckpt ./cifar10_localK.pt
python3 cov_experiment.py --ckpt ./cifar10_localK_sparse.pt

python3 vis.py --ckpt ./mnist_baseline.pt --plot-train-history
python3 vis.py --ckpt ./cifar10_baseline.pt --plot-train-history

python3 vis.py --ckpt ./mnist_globalK.pt --plot-train-history
python3 vis.py --ckpt ./mnist_globalK_sparse.pt --plot-train-history
python3 vis.py --ckpt ./mnist_localK.pt --plot-train-history
python3 vis.py --ckpt ./mnist_localK_sparse.pt --plot-train-history

python3 vis.py --ckpt ./cifar10_globalK.pt --plot-train-history
python3 vis.py --ckpt ./cifar10_globalK_sparse.pt --plot-train-history
python3 vis.py --ckpt ./cifar10_localK.pt --plot-train-history
python3 vis.py --ckpt ./cifar10_localK_sparse.pt --plot-train-history