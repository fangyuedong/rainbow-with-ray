import ray
from collections import namedtuple

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
