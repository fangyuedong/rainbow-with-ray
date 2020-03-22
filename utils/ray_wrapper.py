import ray
from collections import namedtuple

"""wrap a function to remote class"""
@ray.remote
class RayActorWrapFunc():
    def __init__(self, func, *args, **kwargs):
        self.wrap_func = func
        self.args = args

    def __call__(self, **kwargs):
        return self.wrap_func(self.args, **kwargs)

"""wrap a function to remote function"""
def RayFuncWrapFunc(func):
    return ray.remote(func)

RayTsk = namedtuple('RayTsk', ['handle', 'method', 'tsk_ids'])

class RaySche():
    def __init__(self):
        self.name2info = {}
        self.tskid2name = {}

    def add(self, name, actor=None, method=None, **kwargs):
        assert name not in self.name2info.keys()
        if isinstance(actor, ray.actor.ActorHandle) and hasattr(actor, method):
            tsk_ids = tuple(getattr(actor, method).remote(**kwargs))
        elif isinstance(actor, ray.remote_function.RemoteFunction) and method == None:
            tsk_ids = tuple(actor.remote(**kwargs))
        else:
            raise NotImplementedError
        self.name2info[name] = RayTsk(actor, method, tsk_ids)
        for tsk_id in tsk_ids:
            self.tskid2name[tsk_id] = name
        return tsk_ids
    
    def wait(self, timeout=None):
        done_tsk, _ = ray.wait(self.tskid2name.keys(), timeout)
        name = self.tskid2name[done_tsk]
        if len(self.name2info[name].tsk_ids) > 1:
            ray.wait(list(self.name2info[name].tsk_ids), len(self.name2info[name].tsk_ids))
        [self.tskid2name.pop(x) for x in self.name2info[name].tsk_ids]
        return name, self.name2info.pop(name)

    def __len__(self):
        return len(self.name2info)

    def name_list(self):
        return self.name2info.keys()
