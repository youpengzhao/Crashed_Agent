# This is an example to test the model in checkpoint_path under normal cases.
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m_vs_5z checkpoint_path="./results/models/**" evaluate=True save_replay=True final_test=True performance_dir="***" test_nepisode=128
