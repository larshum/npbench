# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
from importlib.metadata import version

from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Any, Callable, Dict, Sequence, Tuple


class TorchFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return version("torch")

    def imports(self) -> Dict[str, Any]:
        import torch
        return {'torch': torch}

    def get_sync_string(self):
        if self.fname.startswith("torch_cuda"):
            return "torch.cuda.synchronize()"
        elif self.fname.startswith("torch_metal"):
            return "torch.mps.synchronize()"
        else:
            raise RuntimeError(f"Unknown framework {self.fname}")

    def copy_func(self) -> Callable:
        """ Returns the copy-method that should be used 
        for copying the benchmark arguments. """

        import numpy as np
        import torch
        if self.fname.startswith("torch_cuda"):
            def copy_func(t):
                t = torch.tensor(t, device='cuda')
                torch.cuda.synchronize()
                return t
            return copy_func
        elif self.fname.startswith("torch_metal"):
            def copy_func(t):
                if t.dtype == np.float64:
                    t = t.astype(np.float32)
                elif t.dtype == np.complex128:
                    t = t.astype(np.complex64)
                t = torch.tensor(t, device='mps')
                torch.mps.synchronize()
                return t
            return copy_func
        return super().copy_func()

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        module_pypath = "npbench.benchmarks.{r}.{m}".format(
                r = bench.info["relative_path"].replace('/', '.'),
                m = bench.info["module_name"])
        if "postfix" in self.info.keys():
            postfix = self.info["postfix"]
        else:
            postfix = self.fname
        module_str = "{m}_{p}".format(m = module_pypath, p = postfix)
        func_str = bench.info["func_name"]

        implementations = []

        # Default implementation
        ldict = dict()
        exec("from {m} import {f} as impl".format(m=module_str, f=func_str), ldict)
        implementations.append((ldict["impl"], 'default'))

        # Implementation with JIT-compilation if it exists
        try:
            exec("from {m} import {f}_jit as impl".format(m=module_str, f=func_str), ldict)
            implementations.append((ldict["impl"], 'jit'))
        except:
            pass

        return implementations

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the setup-string that should be used before calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: The corresponding setup-string.
        """

        sync_str = self.get_sync_string()
        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = ["__npb_copy({})".format(a) for a in bench.info["array_args"]]
            return arg_str + " = " + ", ".join(copy_args) + "; " + sync_str
        return sync_str

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        arg_str = self.arg_str(bench, impl)
        main_exec_str = "__npb_result = __npb_impl({a})".format(a=arg_str)
        sync_str = self.get_sync_string()
        return main_exec_str + "; " + sync_str
