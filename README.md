# Excited state neural network potential using M3GNet model
Ex_MatGL is a [MatGL](https://github.com/materialsvirtuallab/matgl)-based neural network potential that computes excited state energies and forces.

## Installation
1. Install Pytorch. This package is tested on
   - CUDA==11.7
   - Python=3.9.17
   - torch==2.0.0
   ```
   python -m pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117
   ```
2. Install DGL.
   ```
   python -m pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
   python -m pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
   ```
3. Install Matgl

    Clone matgl repository and install latest version.[^1]
    ```
    git clone git@github.com:materialsvirtuallab/matgl.git
    python -m pip install ./matgl
    ```
    [^1]: If you install matgl 0.8.5 version via pip and try to run sample code using GPU, you see the following error.`TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`
4. Install this package
    ```
    python -m pip install .
    ```

## Usage
Sample code is [here](sample/).
Please read [instructuion](sample/README.md).

## References
- C. Chen, S. P. Ong, A universal graph deep learning interatomic potential for the periodic table. [Nature Computational Science. 2, 718–728 (2022)](https://www.nature.com/articles/s43588-022-00349-3).
- https://github.com/materialsvirtuallab/matgl
- Y. Shi, S. Zheng, G. Ke, Y. Shen, J. You, J. He, S. Luo, C. Liu, D. He, T.-Y. Liu, [Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets. arXiv [cs.LG] (2022).](https://arxiv.org/abs/2203.04810)
- https://github.com/microsoft/Graphormer