# Sample programs

## Outline
You have the following directory structure.
```
- alkane
  - train.py
  - evaluate.py
  - visualize.ipynb
- QM5
- acess_heptane
- acess_octane
```
## Run sample code
### Training & Evaluation
Ase database is located at `ex_matgl/DBs`.
First, you make the symbolic link for the database
```
cd alkane
ln -s ../../DBs/alkane.db .
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```
After you perform training, you get the trained model `saved_models/best_model.ckpt`.
After evaluation, you can get the npz file of `evalueate_answer.npz` and `evaluate_prediction.npz`.
Jupyter notebook `visualize.ipynb` show you the results.

### Test for Heptane and Octane
After leaning neural network potential of alkane following the above instructions, run the below codes.
```
cd acess_heptane
ln -s ../../DBs/heptane.db .
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```
You can get the npz file of `heptane_evalueate_answer.npz` and `heptane_evaluate_prediction.npz`.
Jupyter notebook `acess_heptane.ipynb` show you the results.

### Structual Optimization of heptane about S1 
After leaning neural network potential of alkane following the above instructions, run the below codes.
```
cd relax_heptane
ln -s ../../DBs/heptane.db .
python relax.py
```
You can find `optimize_sturctures/bfgs_heptane.traj` and optimized structure of heptane about S1 state in it.