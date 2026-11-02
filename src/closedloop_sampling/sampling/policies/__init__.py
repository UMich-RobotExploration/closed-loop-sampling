"""The five sampling policies (Sec. IV, Strategies), behind one registry.

Registry keys: "static", "random", "lawnmower", "a_optimal", "d_optimal".
Historical aliases from the research records resolve too ("trace",
"dopt", "lawn").
"""
from closedloop_sampling.sampling.policies.base import (
    ALIASES,
    ChainContext,
    CycleContext,
    Policy,
    make_policy,
    policy_names,
)
# Importing the modules registers the policies.
from closedloop_sampling.sampling.policies import (  # noqa: F401
    a_optimal,
    d_optimal,
    lawnmower,
    random_policy,
    static,
)

__all__ = ["Policy", "ChainContext", "CycleContext", "make_policy",
           "policy_names", "ALIASES"]
