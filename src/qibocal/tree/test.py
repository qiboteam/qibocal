import pytest
from execute import Executor
from pending import Queue
from task import Task

# Each task is a list
# [name, parameters, requirements]
tasks = [
    ["command_1", [], ["start"]],
    ["command_2", [], ["command_1"]],
    ["command_3", [], ["command_2", "command_1"]],
]


def test_task():
    job = Task.load(tasks[1])
    assert job.requirements == {"command_1": False}
    assert job.ready == False

    job = Task.load(tasks[0])
    assert job.requirements == {"start": True}
    assert job.ready == True


def test_queue():
    tasks_list = [Task.load(i) for i in tasks]
    q = Queue(tasks_list)
    ready = q.free()

    assert ready[0].id == "command_1"
    assert ready[0].requirements == {"start": True}

    ready = q.free(completed="command_1")
    assert ready[0].id == "command_2"
    assert ready[0].requirements == {"command_1": True}

    ready = q.free(completed="command_2")
    assert ready[0].id == "command_3"
    assert ready[0].requirements == {"command_2": True, "command_1": True}


# def test_executor():

#     executor = Executor.load(tasks)
#     executor.run()
