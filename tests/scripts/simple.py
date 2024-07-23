from qibocal import Executor

executor = Executor.create("mycal")

from mycal import ciao, come

out = ciao(par=1)
res = out if out._results.par[0] > 2 else come(par=42)
