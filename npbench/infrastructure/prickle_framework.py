# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import pkg_resources

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


class PrickleFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return [p.version for p in pkg_resources.working_set if p.project_name.startswith("prickle")][0]

    def imports(self) -> Dict[str, Any]:
        import prickle
        import torch
        return {'prickle': prickle, 'torch': torch}

    def get_target_backend(self):
        if self.fname == "prickle_cuda":
            return "prickle.CompileBackend.Cuda"
        elif self.fname == "prickle_metal":
            return "prickle.CompileBackend.Metal"

    def get_sync_str(self):
        return f"prickle.sync({self.get_target_backend()})"

    def copy_func(self) -> Callable:
        """ Returns the copy-method that should be used 
        for copying the benchmark arguments. """

        import prickle
        import torch
        if self.fname == "prickle_cuda":
            def copy_prickle(t):
                t = torch.tensor(t, device='cuda')
                torch.cuda.synchronize()
                return t
            return copy_prickle
        else:
            def copy_prickle(t):
                t = torch.tensor(t)
                if t.dtype == torch.float64:
                    return t.to(torch.float32)
                if t.dtype == torch.complex128:
                    return t.to(torch.complex64)
                return t
            return copy_prickle

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the setup-string that should be used before calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: The corresponding setup-string.
        """

        sync_str = self.get_sync_str()
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
        sync_str = self.get_sync_str()
        return main_exec_str + "; " + sync_str
