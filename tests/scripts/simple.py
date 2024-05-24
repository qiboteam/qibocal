from qibocal import Executor

executor = Executor.create("mycal")

from mycal import ciao, come

out = ciao(1)
res = out if out > 2 else come(42)
