# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import pkg_resources

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


class Numpy32Framework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return [p.version for p in pkg_resources.working_set if p.project_name.startswith("numpy")][0]

    def imports(self) -> Dict[str, Any]:
        import numpy
        return {'numpy': numpy}

    def copy_func(self) -> Callable:
        """ Returns the copy-method that should be used 
        for copying the benchmark arguments. """

        import numpy as np
        def copy_numpy(t):
            if t.dtype == np.float64:
                return t.astype(np.float32)
            return t
        return copy_numpy
