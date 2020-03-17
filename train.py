from schedule import Sched
from agent import DQN_Worker
from policy_optimizer import Optimizer
from utils.dataloader import Dataloader
from utils.replay_buffer import lmdb_op
import ray
import time
import torch

ray.init()

env_name = "PongNoFrameskip-v4"
n_worker = 1
buffer = "multi_workers_buffer"

lmdb_op.init(buffer)
workers = [ray.remote(DQN_Worker).options(num_gpus=0.1).remote(env_name = "PongNoFrameskip-v4") for _ in range(n_worker)]
dataloader = Dataloader(buffer, lmdb_op, worker_num=2, batch_size=64, batch_num=40)
opt = ray.remote(Optimizer).options(num_gpus=0.3).remote(dataloader, env_name, iter_steps=40, update_period=10000)
sche = Sched()
eps = 1
save_count = 0
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
    sche.wait(len(workers) + 1)

def start():
    [sche.add(worker, "__next__") for worker in workers]

def state_machine(tsk_dones, infos):
    global eps, save_count
    for tsk_done, info in zip(tsk_dones, infos):
        if info.class_name == "DQN_Worker" and info.method == "__next__":
            data, rw = ray.get(tsk_done)
            sche.add(None, lmdb_op.write, lmdb_path=buffer, data=data)
            if rw is not None:
                print("[sche] rw {}".format(rw))
                tsk1 = sche.add(opt, "__call__")
                sche.add(info.handle, "update", state_dict=tsk1, eps=eps)
                save_count += 1
                if save_count == 100:
                    save_count = 0
                    sche.add(info.handle, "save", video_path="./train_video")
            else:
                sche.add(info.handle, "__next__")
        elif info.class_name == "DQN_Worker" and info.method == "update":
            sche.add(info.handle, "__next__")
        elif info.class_name == "DQN_Worker" and info.method == "save":
            pass
        elif info.class_name == None and info.method == lmdb_op.write.__name__:
            if lmdb_op.len(buffer) > 100000 and not sche.have(opt, "__next__"):
                eps = 0.05
                print("[sche] start opt")
                sche.add(opt, "__next__")
        elif info.class_name == None and info.method == lmdb_op.len.__name__:
            pass
        elif info.class_name == "Optimizer" and info.method == "__next__":
            n_step, loss = ray.get(tsk_done)
            print("[sche] loss: {} @ step {}".format(loss, n_step))
            sche.add(info.handle, "__next__")
        elif info.class_name == "Optimizer" and info.method == "__call__":
            pass
        else:
            raise NotImplementedError

def run():
    count = 0
    while 1:
        tsk_dones, infos = sche.wait()
        state_machine(tsk_dones, infos)
        if count % 100 == 0:
            print("[sche] runing tsks {} buff {}".format(len(sche), lmdb_op.len(buffer)))
            count = 0
        count += 1

if __name__ == "__main__":
    init()
    start()
    run()
    

