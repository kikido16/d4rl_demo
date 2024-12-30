# d4rl test of Hopper and Kitchen using d3rlpy
## How to run the code
### Train
Train mujoco model
```
python d3rlpy_demo.py --mode=train --environment=hopper-medium-v0 --algorithm=cql
```
Train kitchen model
```
python d3rlpy_demo.py --mode=train --environment=kitchen-mixed-v0 --algorithm=iql

### Evaluation
Evaluate mujoco model
```
python d3rlpy_demo.py --mode=evaluate --environment=hopper-medium-v0 --model_path=./model/cql_hopper_medium.d3
```
Evaluate kitchen model
```
python d3rlpy_demo.py --mode=evaluate --environment=kitchen-mixed-v1 --model_path=./model/iql_kitchen_mixed_2ksteps_512batch_3.0temp.d3
```