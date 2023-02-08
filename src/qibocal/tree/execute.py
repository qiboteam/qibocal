from dataclasses import dataclass

from history import *
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
    history: History
    outputs: dict[str, Output]

    @classmethod
    def load(cls, card):
        pendings = Queue([Task.load(action) for action in card])
        return cls(
            exe_queue=ExecutionQueue([]),
            pending_queue=pendings,
            history=History([]),
            outputs={},
        )

    def run(self):
        # run all the runnable actions and save them in exe_queue
        self.exe_queue.queue.append(*self.pending_queue.free())

        for wip_task in self.exe_queue:
            # move all the completed actions from exe_queue
            # to history with their outputs
            self.history.record(wip_task[0], wip_task[1])
            # move all the new runnable actions from pending_queue
            # to exe_queue
            self.exe_queue.queue.append(
                self.pending_queue.free(completed=wip_task[0].id)
            )

        if len(self.pending_queue.queue) > 0:
            raise RuntimeError("Execution completed but tasks still pending ")
