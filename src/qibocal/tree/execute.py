from dataclasses import dataclass

from pending import Queue
from task import *


@dataclass
class ExecutionQueue:
    queue: list[Task]

    def __iter__(self):
        wip = self.queue.pop(0)
        yield wip, wip.run()


@dataclass
class Executor:
    exe_queue: ExecutionQueue
    pending_queue: Queue
    # history: History TODO: implement History
    outputs: dict[str, Output]

    @classmethod
    def load(cls, card):
        pendings = Queue([Task.load(action) for action in card])
        return cls(exe_queue=ExecutionQueue([]), pending_queue=pendings, outputs={})

    def run(self):
        self.exe_queue.queue.append(self.pending_queue.free())
        for wip_task in self.exe_queue:  # TODO:solve problem with __iter__
            # self.history.record(wip) TODO: implement History
            self.exe_queue.queue.append(self.pending_queue.free(completed=wip_task.id))
