from queue import Queue

import pytest
from task import Task

# Each task is a list
# [name, parameters, requirements]
tasks = [
    ["command_1", [], ["start"]],
    ["command_2", [], ["command_1"]],
    ["command_3", [], ["command_2", "command_1"]],
]


def test_queue():
    print(tasks)
    list = [Task.load(i) for i in tasks]
    print(type(list[0]))
    q = Queue(list)
    print("prova", q.queue)
    ready = q.free()  # TODO: fix AttributeError: 'Queue' object has no attribute 'free
    assert ready[0].requirements == {"start": True}
    assert ready[0].ready == True


def test_task():
    job = Task.load(tasks[1])
    assert job.requirements == {"command_1": False}
    assert job.ready == False

    job = Task.load(tasks[0])
    assert job.requirements == {"start": True}
    assert job.ready == True
