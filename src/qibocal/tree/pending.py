from dataclasses import dataclass

from task import Task


@dataclass
class Queue:
    queue: list[Task]

    def free(self, completed=None) -> list[Task]:
        ready = []
        for task in self.queue:
            # task.complete(completed) --- What should complete do ?
            if task.ready:
                self.queue.remove(task)
                ready.append(task)
        return ready
