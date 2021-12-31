# This is an example for testing the model under crashed scenarios.
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m_vs_5z checkpoint_path="./results/models/**" evaluate=True save_replay=True final_test=True offline=True performance_dir="***" offline_num=1 test_nepisode=128
