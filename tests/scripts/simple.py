from qibocal import Executor

executor = Executor.create("mycal")

from mycal import ciao, come

out = ciao({"parameters": {"par": 1}})
res = out if out._results.par[0] > 2 else come({"parameters": {"par": 42}})
