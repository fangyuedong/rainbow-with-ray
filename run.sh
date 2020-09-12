# nohup python -u train.py Asterix --num_agents=4 --num_loaders=6 --batch_size=256 --lr=0.625e-4 --suffix="speed_8" --speed=8 >train.txt 2>&1 &

# nohup python -u main.py Asterix --alg=DQN --num_agents=4 --num_loaders=6 --batch_size=256 --lr=0.625e-4 --suffix="DQN_s8_bn2" --speed=8 >train.txt 2>&1 &
nohup python -u main.py Asterix --alg=DDQN  --buffer="pmdb" --num_agents=4 --num_loaders=6 --batch_size=256 --lr=0.15e-4 --suffix="DDQN_pm_0.15" --speed=8 >train.txt 2>&1 &