"""Common function for all submodules"""

import hashlib
from collections.abc import Sequence
import struct
from base64 import b64encode
import numpy as np


def bytes2int(data: bytes) -> int:
    """Convert bytes to an integer"""
    return int.from_bytes(data, "big")


def bytes2str(data: bytes) -> str:
    """Convert bytes into a base64 like string"""
    return b64encode(data).decode().replace("/", "#")


def hash_function(data: bytes) -> bytes:
    """The hash function used to identify similar datasets, methods, configurations, ..
    Returns a bytes object
    """
    return hashlib.sha1(data, usedforsecurity=False).digest()


def hash_function_int(data: bytes) -> int:
    """The hash function used to identify similar datasets, methods, configurations, ..
    Returns an integer
    """
    return bytes2int(hash_function(data))


def hash_function_str(data: bytes) -> str:
    """The hash function used to identify similar datasets, methods, configurations, ..
    Returns an base64 string
    """
    return bytes2str(hash_function(data))


def hash_object_list(objects: Sequence) -> bytes:
    """hash objects in a list
    Will raise a TypeError if an input value has an unsupported type
    """
    type_handling = {
        int: lambda x: hash_function(
            x.to_bytes(length=int((x.bit_length() + 7) / 8), byteorder="big")
        ),
        bytes: hash_function,
        str: lambda x: hash_function(x.encode()),
        list: hash_object_list,
        bool: lambda x: hash_function(bytes(x)),
        float: lambda x: hash_function(struct.pack("d", x)),
        np.ndarray: hash_function,
    }

    hashes = b""
    for value in objects:
        success = False
        for t, handler in type_handling.items():
            if isinstance(value, t):
                hashes += handler(value)
                success = True
                break
        if not success:
            raise TypeError(f"Hashing is not supported for {type(value)}!")
    return hash_function(hashes)


def hash_object_list_int(objects: Sequence) -> int:
    """hash objects in a list and returns an integer
    Will raise a TypeError if an input value has an unsupported type
    """
    return int.from_bytes(hash_object_list(objects), "big")
