import ray
from collections import namedtuple

RayTsk = namedtuple('RayTsk', ['handle', 'method', 'params', 'class_name'])

class Sched():
    def __init__(self):
        self.tsk_info = {}

    def add(self, actor, method="", **kwargs):
        assert isinstance(actor, ray.actor.ActorHandle)
        tsk_id = getattr(actor, method).remote(**kwargs)
        self.tsk_info[tsk_id] = RayTsk(actor, method, kwargs, 
            actor._ray_actor_creation_function_descriptor.class_name)
        print("start\t{}.{}({})".format(actor, method, kwargs))

    def wait(self, num_tsk=1):
        """wait until num of tasks are done"""
        assert len(self.tsk_info) >= num_tsk, \
            "doing tasks num {} is less than require num {}".format(len(self.tsk_info), num_tsk)
        done_tsks, _ = ray.wait(list(self.tsk_info.keys()), num_tsk)
        return done_tsks, [self.tsk_info.pop(tsk_id) for tsk_id in done_tsks]

    

