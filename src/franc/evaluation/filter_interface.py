"""Shared functionality and interface for all filtering techniques"""

from typing import TypeVar, Any, overload
from collections.abc import Sequence, Callable
import abc
from dataclasses import dataclass, asdict, fields
import warnings
import inspect
import functools
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..common import hash_function, hash_object_list

# create a type variable that can be any instance of a Filter subclass
FilterTypeT = TypeVar("FilterTypeT", bound="FilterInterface")


def make_2d_array(A: Sequence | Sequence[Sequence] | NDArray) -> NDArray:
    """add a dimension to 1D arrays and leave 2D arrays as they are
    This is intended to allow 1D array input for single channel application

    :param A: input array

    :return: extended array

    :raises: ValueError if the input shape is not compatible

    >>> import franc as fnc
    >>> fnc.evaluation.make_2d_array([1, 2])
    array([[1, 2]])

    >>> import franc as fnc
    >>> fnc.evaluation.make_2d_array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    """
    A_npy = np.array(A)
    if len(A_npy.shape) == 1:
        return np.array([A_npy])
    if len(A_npy.shape) == 2:
        return A_npy
    raise ValueError("Input must be 1D or 2D array")


def handle_from_dict(init_func: Callable):
    """A decorator for the init functions of classes derived from FitlerInterface

    If the _from_dict keyword argument is passed, the __init__() function is ignored
    and the class is initialized based on the passed dictionary.
    Otherwise, the constructor is called the usual way.
    """
    from_dict_key = "_from_dict"

    @functools.wraps(init_func)
    def wrapper(self, *args, **kwargs):
        if from_dict_key in kwargs and kwargs[from_dict_key] is not None:
            from_dict = kwargs[from_dict_key]
            for key in from_dict:
                setattr(self, key, from_dict[key])
        else:
            # save init parameters
            self.init_parameters = list(args) + [kwargs[key] for key in sorted(kwargs)]

            # calculate method hash
            hashes = self.file_hash()  # pylint: disable=[protected-access]
            hashes += FilterInterface.file_hash()  # pylint: disable=[protected-access]
            hashes += hash_object_list(self.init_parameters)
            self.method_hash_value = hash_function(hashes)

            init_func(self, *args, **kwargs)

    return wrapper


@dataclass
class FilterInterface(abc.ABC):
    """common interface definition for Filter implementations

    :param n_channel: Number of witness sensor channels
    """

    # properties listed here will be export in serialized dumps
    # NOTE: all properties here must be serializable by np.savez()
    #       with allow_pickle=False. This does exclude None!

    # It set to True, a target series is required for this filter type to apply()
    requires_apply_target: bool

    n_channel: int
    method_hash_value: bytes
    supports_multi_sequence = True
    filter_name = "FilterInterface"  # must be implemented in children
    default_args = [None]

    def __init__(self, n_channel: int, _from_dict=None):
        del _from_dict  # make as ignored, it is used by the handle_from_dict decorator
        self.n_channel = n_channel

        if self.__class__ is FilterInterface:
            warnings.warn("Instantiating FilterInterface is not intended!")
        else:
            assert hasattr(
                self, "filter_name"
            ), "BaseFilter childs must declare their name"

        # this can be set to false after the super().__init__() statement in child
        self.requires_apply_target = True

    @staticmethod
    def supports_saving_loading() -> bool:
        """Indicates whether saving and loading is supported
        Due to the way dataclasses work with inheritance, class values with default values don't work in the parent dataclass.
        Thus, this is a function
        """
        return True

    def condition(
        self,
        witness: Sequence | Sequence[Sequence] | NDArray,
        target: Sequence | NDArray,
    ):
        """Use an input dataset to condition the filter

        :param witness: Witness sensor data
        :param target: Target sensor data
        """
        return self.condition_multi_sequence([witness], [target])

    @abc.abstractmethod
    def condition_multi_sequence(
        self,
        witness: Sequence | Sequence[Sequence] | NDArray,
        target: Sequence | NDArray,
    ) -> Any:
        """Similar to condition(), but expects multiple sequences

        First index to the given data objects indicates the sequence.
        The last index indicates the time within a single sequence.
        Sequences must not have the same length.
        """

    def apply(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None = None,
        pad: bool = True,
        update_state: bool = False,
    ) -> NDArray:
        """Apply the filter to a single sequence of input data

        :param witness: Witness sensor data (1D or 2D array)
        :param target: Target sensor data (1D array)
        :param pad: if True, apply padding zeros so that the length matches the target signal
        :param update_state: if True, the filter state will be changed. If false, the filter state will remain

        :return: prediction
        """
        if target is None:
            return self.apply_multi_sequence([witness], None, pad, update_state)[0]
        return self.apply_multi_sequence([witness], [target], pad, update_state)[0]

    @abc.abstractmethod
    def apply_multi_sequence(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None,
        pad: bool = True,
        update_state: bool = False,
    ) -> Sequence[NDArray]:
        """Apply the filter to multiple sequences of input data.

        Similar to apply() but expects multiple sequences.
        First index to the given data objects indicates the sequence.
        The last index indicates the time within a single sequence.
        Sequences must not have the same length.
        """

    def check_data_dimensions(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Check the dimensions of the provided input data and apply make_2d_array()

        :param witness: Witness sensor data
        :param target: Target sensor data

        :return: data as (target, witness)

        :raises: AssertionError
        """
        target_npy = np.array(target)
        witness_npy = make_2d_array(witness)
        assert (
            witness_npy.shape[0] == self.n_channel
        ), "witness data shape does not match configured channel count"

        if self.requires_apply_target:
            assert (
                target is None or target_npy.shape[0] == witness_npy.shape[1]
            ), "Missmatch between target and witness data shapes"

        return witness_npy, target_npy

    # overloads to simplify type handling of return values
    @overload
    def check_data_dimensions_multi_sequence(
        self,
        witness: Sequence | NDArray,
        target: None,
    ) -> tuple[list[NDArray], None]: ...

    @overload
    def check_data_dimensions_multi_sequence(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray,
    ) -> tuple[list[NDArray], list[NDArray]]: ...

    def check_data_dimensions_multi_sequence(
        self,
        witness: Sequence | NDArray,
        target: Sequence | NDArray | None = None,
    ) -> tuple[list[NDArray], list[NDArray] | None]:
        """Check the dimensions of the provided input data and apply make_2d_array()

        :param witness: Witness sensor data
        :param target: Target sensor data

        :return: data as (target, witness)

        :raises: AssertionError
        """
        witness_npy = [make_2d_array(w) for w in witness]
        for w in witness_npy:
            assert (
                w.shape[0] == self.n_channel
            ), "witness data shape does not match configured channel count"

        if target is not None:
            assert len(witness) == len(target)
            target_npy = [np.array(t) for t in target]

            for w, t in zip(witness_npy, target_npy):
                assert (
                    w.shape[1] == t.shape[0]
                ), "Missmatch between target and witness data shapes"
        else:
            target_npy = None

        if self.requires_apply_target:
            assert (
                target is not None
            ), "This filter requires a target signal to be applied"

        return witness_npy, target_npy

    def as_dict(self) -> dict[str, Any]:
        """Returns a dictionary that represents the state of this filter."""
        if not self.supports_saving_loading():
            raise NotImplementedError(
                "Saving and loading is not supported for this filter type."
            )

        return asdict(self)

    @classmethod
    def from_dict(cls: type[FilterTypeT], input_dict: dict[str, Any]) -> FilterTypeT:
        """Create a filter instance from a dictionary that was created from as_dict()"""
        # check that the dict contains all relevant keys and eliminate any extra keys
        if not cls.supports_saving_loading():
            raise NotImplementedError(
                "Saving and loading is not supported for this filter type."
            )

        try:
            clean_dict = {key.name: input_dict[key.name] for key in fields(cls)}
        except Exception as e:
            raise ValueError("Non-compatible dictionary, could not load filter.") from e

        if not hasattr(cls, "filter_name"):
            raise ValueError("From_dict cannot be used on an unnamed filter class.")
        if cls.filter_name != clean_dict["filter_name"]:  # pylint: disable=no-member
            raise ValueError(
                f'Loading a {clean_dict["filter_name"]} as {cls.filter_name} is not possible.'  # pylint: disable=no-member
            )

        # passing the _from_dict value should make all other values be relevant
        # They are set to the incompatible None type to potentially fail early in case of an error
        return cls(*cls.default_args, _from_dict=clean_dict)  # type: ignore[arg-type, misc]

    @classmethod
    def make_filename(cls: type[FilterTypeT], filename: str | Path):
        """Append the file type of save files for this class to the given filename, if it is not already present"""
        if isinstance(filename, Path):
            filename = str(filename)
        ending = "." + cls.filter_name + ".npz"  # pylint: disable=no-member
        if not filename.endswith(ending):
            filename += ending
        return filename

    def save(self, filename: str | Path, warn_incompatible: bool = False):
        """Save the filter state as a numpy file

        The given filename will be autocompleted with a  ".<filter_name>.npz"
        filename extension, unless a matching extension is detected.

        warn_incompatible: set to True to warn for object types might not
                           compatible with np.save(allow_pickle=False) during development
        """
        serialization_data = self.as_dict()
        filename = self.make_filename(filename)

        # this is intended to quickly identify problematic values when developing a new filter
        if warn_incompatible:
            for k, v in serialization_data.items():
                if type(v) not in {str, int, float, np.ndarray, bool}:
                    print(
                        f">> Potentially incompatible with numpy.save(pickle=False) {k}: {v}"
                    )

        # pickles are disable for security reasons
        np.savez(filename, allow_pickle=False, **serialization_data)

    @classmethod
    def load(cls: type[FilterTypeT], filename) -> FilterTypeT:
        """Load a filter state from the supplied filename.

        The given filename will be autocompleted with a  ".<filter_name>.npz"
        filename extension, unless a matching extension is detected.
        """
        if not cls.supports_saving_loading():
            raise NotImplementedError(
                "Saving and loading is not supported for this filter type."
            )

        filename = cls.make_filename(filename)

        # pickles are disable for security reasons
        filter_dict = dict(np.load(filename, allow_pickle=False))
        return cls.from_dict(filter_dict)

    @classmethod
    def file_hash(cls: type[FilterTypeT]) -> bytes:
        """Calculates a hash value based on the file in which this method was defined."""
        try:
            with open(inspect.getfile(cls), "rb") as f:
                script = f.read()
        except TypeError:
            try:
                script = inspect.getsource(cls).encode()
            except TypeError:
                script = cls.filter_name.encode()
                warnings.warn(
                    f"Could not include source code in hash for {cls.filter_name}"
                )
        return hash_function(script)

    @property
    def method_hash(self) -> bytes:
        """A hash of the method and parameters
        NOTE: This is not a hash of the conditioned filter!
        Thus, the same filter configuration applied to a different dataset will result in the same hash!
        """
        return self.method_hash_value

    @property
    def method_filename_part(self) -> str:
        """string that can be used in a file name"""
        return f"{self.filter_name}_{self.n_channel}_"
