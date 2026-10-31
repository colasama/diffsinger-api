"""Bound in-flight jobs and share model stages across them in one server process.

Waiting jobs hold no model-stage slot. Acquire a stage before the GPU budget;
callers must not nest resource contexts. FIFO gates prevent phrase-sized
synthesis calls from starving a waiting separation or alignment request.
"""

from collections import deque
from contextlib import contextmanager
from functools import wraps
from threading import Condition, Lock

from . import settings


class Gate:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.active = 0
        self.waiters = deque()
        self.condition = Condition()

    @contextmanager
    def enter(self):
        ticket = object()
        with self.condition:
            self.waiters.append(ticket)
            try:
                self.condition.wait_for(lambda: self.waiters[0] is ticket and self.active < self.capacity)
            except BaseException:
                self.waiters.remove(ticket)
                self.condition.notify_all()
                raise
            self.waiters.popleft()
            self.active += 1
            self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def status(self):
        with self.condition:
            return {"capacity": self.capacity, "active": self.active, "waiting": len(self.waiters)}


class Scheduler:
    def __init__(self, active_jobs: int, gpu_concurrency: int):
        self.jobs = Gate(active_jobs)
        self.gpu = Gate(gpu_concurrency)
        self._stages: dict[str, Gate] = {}
        self._lock = Lock()

    @contextmanager
    def resource(self, name: str, *, gpu: bool = False):
        with self._lock:
            stage = self._stages.setdefault(name, Gate(1))
        with stage.enter():
            if gpu:
                with self.gpu.enter():
                    yield
            else:
                yield

    def status(self):
        with self._lock:
            stages = dict(self._stages)
        return {"jobs": self.jobs.status(), "gpu": self.gpu.status(),
                "stages": {name: gate.status() for name, gate in stages.items()}}


scheduler = Scheduler(settings.ACTIVE_JOBS, settings.GPU_CONCURRENCY)


def stage(name: str, *, gpu: bool = False):
    """Protect the complete use of mutable model state, including cold loading."""
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            with scheduler.resource(name, gpu=gpu):
                return function(*args, **kwargs)
        return run
    return decorate
