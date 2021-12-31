```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
```
# Crashed_agents
- This work is based on the Pymarl library, so many points are the same as Pymarl. If something, which is not relevant with our adaptive framework, is not clear, please refer to Pyamrl.  We recommend you build a Pymarl environment first and refer to our code to make accordingly modifications.
- The modifications of our work are mainly in these files: "/src/config/default.yaml", "/src/runners/episode_runners.py", "/src/run.py", "/src/learners/q_learner.py" and "/src/controllers/basic_controller.py". To be mentioned, in order to adopt our method in QPLEX, please use the open-source code offered by its authors. Then you can make corresponding modifications like our method.

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recommended).

### Creating new maps
We design some new maps during the experiment. Users can extend SMAC by adding new maps/scenarios. To this end, one needs to:

- Design a new map/scenario using StarCraft II Editor:
  - Please take a close look at the existing maps to understand the basics that we use (e.g. Triggers, Units, etc),
  - [Here](https://docs.google.com/document/d/1BfAM_AtZWBRhUiOBcMkb_uK4DAZW3CpvO79-vnEOKxA/edit?usp=sharing) is the step-by-step guide on how to create new RL units based on existing SC2 units,
- Add the map information in [smac_maps.py](https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/maps/smac_maps.py), which is in the installed SMAC library if the environment is installed correctly.
- The newly designed RL units have new ids which need to be handled in [starcraft2.py](https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/starcraft2.py). Specifically, for heterogenious maps containing more than one unit types, one needs to manually set the unit ids in the `_init_ally_unit_types()` function.


## Run an experiment 

```shell
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z use_tensorboard=True save_model=True
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

The config parameters that we add in the experiment is located in `src/config/default.yaml` so that our method can be used in all the algorithms that Pymarl supports. If you want to use our adaptive framework, you just need to add some parameters in the command just as the above command. The meaning of the parameters is explained in `default.yaml`.

All results will be stored in the `Results` folder.

The previous config files used for the SMAC Beta have the suffix `_beta`.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 


### Testing the models

We use two shell to test the performance of the model, which are called "final_test.sh" and "offline.sh". Note that the shell files only serve as examples. To test the performance, you have to change the command corresponding to the map that you use. We record the result of the performance in a txt file, whose path is also an args parameter. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them if StarCraft II client is installed.

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

