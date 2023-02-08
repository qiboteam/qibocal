import copy
from dataclasses import dataclass

from task import Task


@dataclass
class Queue:
    queue: list[Task]

    def free(self, completed=None) -> list[Task]:
        # this function removes all the runnable actions from the queue
        # and saves them in ready
        ready = []
        j = 0
        while j < len(self.queue):
            task = self.queue[j]
            task.complete(completed)
            if task.ready:
                self.queue.remove(task)
                ready.append(task)
            else:
                j += 1
        return ready
