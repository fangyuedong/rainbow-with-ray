from schedule import Sched
from agent import DQN_Worker
from policy_optimizer import DDQN_Opt as Optimizer
from utils.dataloader import Dataloader
from utils.replay_buffer import mmdb_op as lmdb_op
import ray
import time
import torch
from tensorboardX import SummaryWriter

n_worker = 1
n_iter = 40
n_loader = 2
env_name = "AsterixNoFrameskip-v4"
buffer = "multi_workers_buffer"
batch_size = 32
lr = 2.5e-4
ray.init(num_cpus=1+n_worker+n_loader, object_store_memory=1*1024**3, memory=6*1024**3)

buffer = lmdb_op.init(buffer)
workers = [ray.remote(DQN_Worker).options(num_gpus=0.1).remote(env_name=env_name, db=buffer, db_write=lmdb_op.write) for _ in range(n_worker)]
test_worker = ray.remote(DQN_Worker).options(num_gpus=0.1).remote(env_name=env_name, phase="test", suffix="Adam_32")
worker_id = {worker: "worker_{}".format(i) for i, worker in enumerate(workers)}
dataloader = Dataloader(buffer, lmdb_op, worker_num=n_loader, batch_size=batch_size, batch_num=n_iter)
opt = ray.remote(Optimizer).options(num_gpus=0.3).remote(dataloader, env_name, suffix="Adam_32", iter_steps=n_iter, update_period=10000, lr=lr)
sche = Sched()
eps = 1
opt_start = False
glog = SummaryWriter("./logdir/{}/{}.lr{}.batch{}".format(env_name, Optimizer.__name__, lr, batch_size), filename_suffix=env_name)
model_save_period = 100000
train_step = 0
model_idx = 0
total_envs_steps = 0
curr_train_steps = 0

"""
class_name          method
DQN_Worker          __next__
DQN_Worker          update
DQN_Worker          save
None                lmdb_op.write()
None                lmdb_op.len()
Optimizer           __next__
Optimizer           __call__
"""

def init():
    tsk_id = sche.add(opt, "__call__")
    [sche.add(worker, "update", state_dict=tsk_id, eps=eps) for worker in workers]

def start():
    [sche.add(worker, "__next__") for worker in workers]

def state_machine(tsk_dones, infos):
    global eps, opt_start, train_step, model_idx, total_envs_steps, curr_train_steps
    if lmdb_op.len(buffer) > 100000 and opt_start == False:
        eps = 0.1
        print("[sche] start opt")
        sche.add(opt, "__next__")
        opt_start = True
    for tsk_done, info in zip(tsk_dones, infos):
        if info.handle in workers and info.method == "__next__":
            wk_info = ray.get(tsk_done)
            tsk1 = sche.add(opt, "__call__")
            sche.add(info.handle, "update", state_dict=tsk1, eps=eps)
            print("[sche] rw {}".format(wk_info["episod_rw"]))
            glog.add_scalar("rw/{}".format(worker_id[info.handle]), wk_info["episod_rw"], wk_info["total_env_steps"])
            glog.add_scalar("real_rw/{}".format(worker_id[info.handle]), wk_info["episod_real_rw"], wk_info["total_env_steps"])
            total_envs_steps += wk_info["total_env_steps"]
            if opt_start and 8 * total_envs_steps > curr_train_steps*batch_size:
                sche.add(opt, "__next__")
        elif info.handle in workers and info.method == "update":
            sche.add(info.handle, "__next__")
        elif info.class_name == None and info.method == lmdb_op.len.__name__:
            pass
        elif info.class_name == Optimizer.__name__ and info.method == "__next__":
            opt_info = ray.get(tsk_done)
            curr_train_steps += opt_info["opt_steps"]
            if opt_info["opt_steps"] >= train_step + model_save_period:
                sche.add(info.handle, "save")
                train_step = opt_info["opt_steps"]
            print("[sche] loss: {} @ step {}".format(opt_info["loss"], opt_info["opt_steps"]))
            glog.add_scalar("loss", opt_info["loss"], opt_info["opt_steps"])
        elif info.class_name == Optimizer.__name__ and info.method == "save":
            path = ray.get(tsk_done)
            sche.add(test_worker, "load", path=path)
            sche.add(test_worker, "__next__")
        elif info.class_name == Optimizer.__name__ and info.method == "__call__":
            pass
        elif info.handle == test_worker and info.method == "load":
            pass
        elif info.handle == test_worker and info.method == "__next__":
            test_rw = ray.get(tsk_done)
            glog.add_scalar("test_rw", test_rw, model_idx)
            model_idx += 1
        else:
            raise NotImplementedError

def run():
    count, iters, t0 = 0, 0, time.time()
    while 1:
        tsk_dones, infos = sche.wait()
        state_machine(tsk_dones, infos)
        if infos[0].class_name == Optimizer.__name__ and infos[0].method == "__next__":
            iters += 1
            if iters == 100:
                t1 = time.time()
                print("[sche] iter speed: {}/s".format(100*n_iter/(t1-t0)))
                t0, iters = t1, 0
        if count % 100 == 0:
            print("[sche] runing tsks {} buff {}".format(len(sche), lmdb_op.len(buffer)))
            count = 0
        count += 1

if __name__ == "__main__":
    init()
    start()
    run()
    

