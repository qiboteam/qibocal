from enum import Enum


def command_1():
    print("command 1")


def command_2():
    print("command 2")


def command_3():
    print("command 3")


class Operation(Enum):
    command_1 = command_1  # TODO: this is not working
    command_2 = command_2
    command_3 = command_3
