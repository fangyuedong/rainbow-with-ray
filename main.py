from schedule import Sched, Engine
from agent import DQN_Worker, PriorDQN_Worker
import policy_optimizer #import DDQN_Opt as Optimizer
from utils.dataloader import Dataloader
import utils.replay_buffer #import mmdb_op as lmdb_op
import ray
import time
import torch
from tensorboardX import SummaryWriter
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("env_name", default="Pong", help="atari env name")
parse.add_argument("--test", action='store_true', default=False, help="only do test")
parse.add_argument("--alg", default="DDQN", help="optimizer algorithm(DQN, DDQN)")
parse.add_argument("--buffer", default="mmdb", help="replay buffer type(mmdb, pmdb)")
parse.add_argument("--num_agents", default=1, type=int, help="num of agents")
parse.add_argument("--num_loaders", default=2, type=int, help="num of data load processing")
parse.add_argument("--batch_size", default=256, type=int, help="train batch size")
parse.add_argument("--lr", default=0.625e-4, type=float, help="learning rate")
parse.add_argument("--suffix", default="default", help="suffix for saving folders")
parse.add_argument("--speed", default=8, type=int, help="training data consum speed. default 8x data generate speed")
parse.add_argument("--resume", default=None, type=str, help="path to pretrain model")
args = parse.parse_args()

if args.test == False:
    Optimizer = policy_optimizer.__dict__["{}_Opt".format(args.alg)]
    lmdb_op = utils.replay_buffer.__dict__["{}_op".format(args.buffer)]

    n_worker = args.num_agents
    n_loader = args.num_loaders
    env_name = "{}NoFrameskip-v4".format(args.env_name)
    buffer = ""
    batch_size = args.batch_size
    lr = args.lr
    suffix = args.suffix
    speed = args.speed
    n_iter = 40*32//batch_size
    write_prior = (args.buffer == "pmdb")
    ray.init(num_cpus=1+2*n_worker+n_loader, object_store_memory=4*1024**3, memory=12*1024**3)

    buffer = lmdb_op.init(buffer, alpha=0.5)
    workers = [ray.remote(PriorDQN_Worker).options(num_gpus=0.1).remote(env_name=env_name, db=buffer, db_write=lmdb_op.write) for _ in range(n_worker)]
    test_worker = ray.remote(DQN_Worker).options(num_gpus=0.1).remote(env_name=env_name, phase="valid", suffix=suffix)
    dataloader = Dataloader(buffer, lmdb_op, worker_num=n_loader, batch_size=batch_size, batch_num=n_iter)
    opt = ray.remote(Optimizer).options(num_gpus=0.2).remote(dataloader, env_name, suffix=suffix, iter_steps=n_iter, update_period=10000, lr=lr)
    glog = SummaryWriter("./logdir/{}/{}/{}.lr{}.batch{}".format(env_name, suffix, Optimizer.__name__, lr, batch_size))

    engine = Engine(opt, workers, test_worker, buffer, glog, speed)

    engine.reset()
    while engine.stop():
        engine.step()
else:
    env_name = "{}NoFrameskip-v4".format(args.env_name)
    suffix = args.suffix
    worker = DQN_Worker(env_name=env_name, phase="test", suffix=suffix)
    worker.load(args.resume)
    worker.update(eps=0.05)
    score = next(worker)
    print(score)




    

