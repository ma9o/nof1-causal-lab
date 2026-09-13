"""JSON/Pydantic boundary for native NumPyro distributions and their constructor trees.

The Python field is a NumPyro Distribution. JSON records native constructors,
arrays, transforms, and constraints; no probability behavior is implemented here.
"""

from __future__ import annotations

import inspect
from typing import Annotated, Any

import jax
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from pydantic import GetPydanticSchema
from pydantic_core import core_schema

from nof1_causal_lab.json_types import JsonValue

_NATIVE_MODULES = {
    "distribution": (dist, dist.Distribution),
    "transform": (transforms, transforms.Transform),
    "constraint": (constraints, constraints.Constraint),
}


def _constructor_arguments(value: object) -> dict[str, object]:
    cls = type(value)
    arguments = value.get_args() if isinstance(value, dist.Distribution) else {}
    # NumPyro caches equivalent matrix parameterizations. A constructor accepts
    # just one; its stored Cholesky factor preserves the native law directly.
    if "scale_tril" in arguments:
        arguments.pop("covariance_matrix", None)
        arguments.pop("precision_matrix", None)
    aliases = {"base_distribution": "base_dist", "mask": "_mask", "transform": "_inv"}
    for name, parameter in inspect.signature(cls).parameters.items():
        if name in arguments or name == "validate_args":
            continue
        if name in getattr(cls, "arg_constraints", {}):
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        attribute = aliases.get(name, name)
        if hasattr(value, attribute):
            arguments[name] = getattr(value, attribute)
        elif parameter.default is inspect.Parameter.empty:
            raise ValueError(f"Cannot serialize native constructor argument {cls.__name__}.{name}")
    return arguments


def _encode(value: object) -> JsonValue:
    for tag, (module, base) in _NATIVE_MODULES.items():
        if isinstance(value, base):
            cls = type(value)
            if getattr(module, cls.__name__, None) is not cls:
                raise ValueError(f"{cls.__name__} is not a native NumPyro {tag} constructor")
            return {
                tag: cls.__name__,
                "params": {
                    name: _encode(item) for name, item in _constructor_arguments(value).items()
                },
            }
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError("Complex distribution arguments are not JSON values")
        return {"array": _encode(array.tolist()), "dtype": str(array.dtype)}
    if isinstance(value, tuple):
        return {"tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return {"float": str(value)}
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise ValueError(f"Cannot serialize native distribution argument {type(value).__name__}")


def _decode(value: JsonValue) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    for tag, (module, base) in _NATIVE_MODULES.items():
        if tag in value:
            cls = getattr(module, str(value[tag]), None)
            if not isinstance(cls, type) or not issubclass(cls, base):
                raise ValueError(f"Unknown native NumPyro {tag} constructor {value[tag]!r}")
            if set(value) != {tag, "params"} or not isinstance(value["params"], dict):
                raise ValueError(f"A native {tag} requires exactly its name and constructor params")
            arguments = {name: _decode(item) for name, item in value["params"].items()}
            if "validate_args" in inspect.signature(cls).parameters:
                arguments["validate_args"] = True
            try:
                return cls(**arguments)
            except TypeError as exc:
                raise ValueError(f"Invalid {cls.__name__} constructor arguments: {exc}") from exc
    if set(value) == {"array", "dtype"}:
        return np.asarray(_decode(value["array"]), dtype=str(value["dtype"]))
    if set(value) == {"tuple"} and isinstance(value["tuple"], list):
        return tuple(_decode(item) for item in value["tuple"])
    if set(value) == {"float"} and value["float"] in {"inf", "-inf", "nan"}:
        return float(str(value["float"]))
    return {name: _decode(item) for name, item in value.items()}


def encode_distribution(value: dist.Distribution) -> dict[str, JsonValue]:
    """Serialize a native law without flattening batch/event dimensions or mixtures."""
    encoded = _encode(value)
    assert isinstance(encoded, dict)
    return encoded


def decode_distribution(value: dict[str, JsonValue]) -> dist.Distribution:
    """Restore and validate the exact native constructor tree."""
    decoded = _decode(value)
    if not isinstance(decoded, dist.Distribution):
        raise ValueError("Expected a NumPyro distribution constructor")
    return decoded


def _validate_distribution(value: dist.Distribution) -> dist.Distribution:
    value.validate_args()
    return value


def _distribution_schema(_source, handler):
    wire = core_schema.typed_dict_schema(
        {
            "distribution": core_schema.typed_dict_field(core_schema.str_schema()),
            "params": core_schema.typed_dict_field(handler.generate_schema(dict[str, JsonValue])),
        },
        extra_behavior="forbid",
        ref="NumPyroDistribution",
    )
    decoder = core_schema.no_info_after_validator_function(decode_distribution, wire)
    return core_schema.json_or_python_schema(
        json_schema=decoder,
        python_schema=core_schema.union_schema(
            [
                core_schema.no_info_after_validator_function(
                    _validate_distribution, core_schema.is_instance_schema(dist.Distribution)
                ),
                decoder,
            ]
        ),
        serialization=core_schema.plain_serializer_function_ser_schema(
            encode_distribution, when_used="json", return_schema=wire
        ),
    )


NumPyroDistribution = Annotated[dist.Distribution, GetPydanticSchema(_distribution_schema)]
