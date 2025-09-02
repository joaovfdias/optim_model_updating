import json, gzip, importlib, time, random
from typing import Any, Dict, List, Optional
import numpy as np

_JSON_NUMPY = (np.integer, np.floating)
def _to_py(v):
    if isinstance(v, _JSON_NUMPY): return v.item()
    if isinstance(v, np.ndarray):  return v.tolist()
    return v

def dumps_json(obj: Any, path: str) -> None:
    # .json.gz -> gzip
    data = json.dumps(obj, ensure_ascii=False, indent=2, default=_to_py)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f: f.write(data)
    else:
        with open(path, "w", encoding="utf-8") as f: f.write(data)

def loads_json(path: str) -> Any:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f: return json.load(f)
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

# Parameter <-> dict
def param_to_dict(p) -> Dict[str, Any]:
    kind = p.__class__.__name__
    d = {"kind": kind, "key": p.key}
    if kind in ("Continuous", "State"):
        d["lower"] = p.lower_bound
        d["upper"] = p.upper_bound
    return d

def param_from_dict(d):
    kind = d["kind"]
    if kind == "Continuous":
        from optimization.parameter import Continuous
        return Continuous(d["lower"], d["upper"], d["key"])
    if kind == "Binary":
        from optimization.parameter import Binary
        return Binary(d["key"])
    if kind == "State":
        from optimization.parameter import State
        return State(d["lower"], d["upper"], d["key"])
    raise ValueError(f"Unknown Parameter kind: {kind}")

# Individual/Particle <-> dict
def indiv_to_dict(ind) -> Dict[str, Any]:
    base = {
        "type": ind.__class__.__name__,
        "param": [float(x) for x in ind.param],
        "fitness": None if ind.fitness is None else float(ind.fitness),
        "data": ind.data,
        "etime": float(ind.etime),
    }
    # PSO particle extras
    if base["type"] == "Particle":
        base["pso"] = {
            "velocity": None if getattr(ind, "velocity", None) is None else [float(v) for v in ind.velocity],
            "best": None if getattr(ind, "best", None) is None else {
                "param": [float(x) for x in ind.best[0]], "fitness": float(ind.best[1])
            }
        }
    return base

def indiv_from_dict(d, fitness_function):
    t = d.get("type", "Individual")
    if t == "Particle":
        from optimization.pso_optimizer.particle import Particle
        vel = None
        best = None
        if d.get("pso"):
            vel = d["pso"].get("velocity")
            b = d["pso"].get("best")
            if b:
                best = [b["param"], b["fitness"]]
        ind = Particle(d["param"], fitness_function, vel, best)
    else:
        from optimization.individual import Individual
        ind = Individual(d["param"], fitness_function)

    ind.fitness = d.get("fitness")
    ind.data    = d.get("data")
    ind.etime   = d.get("etime")
    return ind


# Fitness spec loader (opcional)
def load_fitness_from_spec(spec: Optional[Dict[str, Any]]):
    if not spec: return None
    t = spec.get("type")
    if t == "callable_path":
        mod, func = spec["path"].split(":")
        return getattr(importlib.import_module(mod), func)
    # "script_path" e "external_spec" podem ser tratados aqui conforme seu projeto
    return None
