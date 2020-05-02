import ray
from collections import namedtuple
import time

RayTsk = namedtuple('RayTsk', ['handle', 'method', 'params', 'class_name'])

class Sched():
    def __init__(self):
        self.tsk_info = {}
        self.remote_func = {}

    def add(self, actor=None, method=None, **kwargs):
        # exec some class method
        if isinstance(actor, ray.actor.ActorHandle) and hasattr(actor, method):
            tsk_id = getattr(actor, method).remote(**kwargs)
            self.tsk_info[tsk_id] = RayTsk(actor, method, None, 
                actor._ray_actor_creation_function_descriptor.class_name)
        # exec some func
        elif callable(method) and actor is None:
            if method not in self.remote_func:
                self.remote_func[method] = ray.remote(method)
            tsk_id = self.remote_func[method].remote(**kwargs)
            self.tsk_info[tsk_id] = RayTsk(actor, method.__name__, None, None)
        else:
            raise NotImplementedError
        return tsk_id

    def wait(self, num_tsk=1, timeout=None):
        """wait until num of tasks are done"""
        assert len(self.tsk_info) >= num_tsk, \
            "doing tasks num {} is less than require num {}".format(len(self.tsk_info), num_tsk)
        done_tsks, _ = ray.wait(list(self.tsk_info.keys()), num_tsk, timeout)
        return done_tsks, [self.tsk_info.pop(tsk_id) for tsk_id in done_tsks]

    def __len__(self):
        return len(self.tsk_info)

    def have(self, actor=None, method=None):
        if isinstance(actor, ray.actor.ActorHandle) and hasattr(actor, method):
            pass
        elif callable(method) and actor is None:
            method = method.__name__
        else:
            raise NotImplementedError
        for _, item in self.tsk_info.items():
            if item.handle == actor and item.method == method:
                return True
        else:
            return False


class Engine():
    def __init__(self, opt, exec_workers, test_worker, replay_buff, glog, speed=8, update_period=10, replay_start=100000):
        assert isinstance(opt, ray.actor.ActorHandle)
        assert isinstance(test_worker, ray.actor.ActorHandle)
        assert isinstance(replay_buff, ray.actor.ActorHandle)
        for worker in exec_workers: assert isinstance(worker, ray.actor.ActorHandle) 
        self.opt = opt
        self.exec_workers = exec_workers
        self.worker_id = {worker: "worker_{}".format(i) for i, worker in enumerate(self.exec_workers)}
        self.test_worker = test_worker
        self.replay_buff = replay_buff
        self.glog = glog
        self.s = speed
        self.replay_start = replay_start
        self.update_period = update_period
        self.batch_size = ray.get(self.opt.config.remote())["batch_size"]

        self.opt_steps = 0
        self.opt_steps_exec = 0
        self.opt_steps_test = 0
        self.env_steps = 0
        self.model_idx = 0
        self.newest_p = None
        self.opting = False

        self.sche = Sched()

    def get_eps(self):
        if self.env_steps < self.replay_start:
            eps = 1.0
        else:
            eps = 0.1
        return eps
    
    def get_exec_worker_param(self):
        if self.opt_steps - self.opt_steps_exec > self.update_period:
            self.newest_p = self.sche.add(self.opt, "__call__")
            self.opt_steps_exec = self.opt_steps
        return self.newest_p

    def add_opt(self):
        if (not self.sche.have(self.opt, "__next__")) and (self.env_steps - self.replay_start) * self.s > self.opt_steps * self.batch_size:
            self.sche.add(self.opt, "__next__")

    def add_exec_work(self, handle):
        p, eps = self.get_exec_worker_param(), self.get_eps()
        self.sche.add(handle, "update", state_dict=p, eps=eps)
        self.sche.add(handle, "__next__")

    def add_test_work(self):
        if (not self.sche.have(self.test_worker, "__next__")) and self.opt_steps // 100000 - self.opt_steps_test // 100000 == 1:
            tsk_id = self.sche.add(self.opt, "save")
            self.sche.add(self.test_worker, "load", path=tsk_id)
            self.sche.add(self.test_worker, "__next__")
            self.opt_steps_test = self.opt_steps

    def check_exec_worker_done(self, tsk_info, tsk_id):
        if tsk_info.handle in self.exec_workers and tsk_info.method == "__next__":
            info = ray.get(tsk_id)
            print("[sche] rw {}".format(info["episod_rw"]))
            self.glog.add_scalar("rw/{}".format(self.worker_id[tsk_info.handle]), info["episod_rw"], info["total_env_steps"])
            self.glog.add_scalar("real_rw/{}".format(self.worker_id[tsk_info.handle]), info["episod_real_rw"], info["total_env_steps"])
            self.env_steps += info["episod_len"]  
            # print(self.env_steps)  
            return True
        return False        

    def check_opt_done(self, tsk_info, tsk_id):
        if tsk_info.handle == self.opt and tsk_info.method == "__next__":
            info = ray.get(tsk_id)
            print("[sche] loss: {} @ step {}".format(info["loss"], info["opt_steps"]))
            self.glog.add_scalar("loss", info["loss"], info["opt_steps"]) 
            self.opt_steps = info["opt_steps"]  
            return True
        return False      

    def check_test_work_done(self, tsk_info, tsk_id):
        if tsk_info.handle == self.test_worker and tsk_info.method == "__next__":
            info = ray.get(tsk_id)
            self.glog.add_scalar("test_rw", info, self.model_idx)
            self.model_idx += 1  
            return True
        return False          

    def step(self):
        tsk_dones, infos = self.sche.wait()
        tsk_done, info = tsk_dones[0], infos[0]
        if self.check_exec_worker_done(info, tsk_done):
            self.add_exec_work(info.handle)
        self.check_test_work_done(info, tsk_done)
        self.check_opt_done(info, tsk_done)

        self.add_test_work()
        self.add_opt() if self.opting == True else None

        self.opting = self.env_steps > self.replay_start

    def reset(self):
        tsk_id = self.sche.add(self.opt, "__call__")
        eps = self.get_eps()
        [self.sche.add(worker, "update", state_dict=tsk_id, eps=eps) for worker in self.exec_workers]
        [self.sche.add(worker, "__next__") for worker in self.exec_workers]
        