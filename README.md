# d4rl test of Hopper and Kitchen using d3rlpy
## How to run the code
Dependencies:
- d3rlpy-2.6.0

### Train
Train mujoco model
```
python d3rlpy_demo.py --mode=train --environment=hopper-medium-v0 --algorithm=cql
```
Train kitchen model
```
python d3rlpy_demo.py --mode=train --environment=kitchen-mixed-v1 --algorithm=iql
```

```
python d3rlpy_demo.py --mode=train --environment=D4RL/kitchen/complete-v2 --algorithm=bc
```

### Evaluation
Evaluate mujoco model
```
python d3rlpy_demo.py --mode=evaluate --environment=hopper-medium-v0 --model_path=./model/cql_hopper_medium.d3
```
Evaluate kitchen model
```
python d3rlpy_demo.py --mode=evaluate --environment=kitchen-mixed-v1 --model_path=./model/iql_kitchen_mixed_2ksteps_512batch_3.0temp.d3
```

```
python d3rlpy_demo.py --mode=evaluate --environment=kitchen-complete-v1 --model_path=./model/bc_kitchen_complete_1e-3lrate_128batch_2ksteps.d3

```

### Trained results

#### Hopper
orange curve: Hopper-expert-v0, conservative weight=5.0, Encoder=[256, 256, 256]

red curve: Hopper-medium-v0, conservative weight=10.0, Encoder=[256, 256, 256]

blue curve: Hopper-medium-v0, conservative weight=10.0, NoEncoder

black curve: Hopper-medium-v0, conservative weight=5.0, Encoder=[256, 256, 256]
![image](https://github.com/kikido16/d4rl_demo/blob/master/visualization/Hopper.png)

evaluation video:

![image](https://github.com/kikido16/d4rl_demo/blob/master/video/hopper_test.mp4/rl-video-episode-0.gif)

#### Kitchen
kitchen-mixed-v0, IQL

orange curve: batch_size=1024, Inverse temperature=3.0, Encoder=[512, 512, 512]

green curve: batch_size=512, Inverse temperature=3.0, Encoder=[512, 512, 512]

pink curve: batch_size=256, Inverse temperature=0.5, Encoder=[256, 256]

black curve: batch_size=4096, Inverse temperature=0.5, Encoder=[256, 256]

![image](https://github.com/kikido16/d4rl_demo/blob/master/visualization/kitchen_mixed_iql.png)