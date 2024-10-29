"""Internal module use at your own risk

This module provides a minimal interface for working with ggml tensors from bark-cpp-python
"""
import os
import pathlib
import ctypes
from typing import NewType


import bark_cpp._ctypes_extensions as ctypes_ext

libggml_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib"
libggml = ctypes_ext.load_shared_library("ggml", libggml_base_path)

ctypes_function = ctypes_ext.ctypes_function_for_shared_library(libggml)


# struct ggml_init_params {
#     // memory pool
#     size_t mem_size;   // bytes
#     void * mem_buffer; // if NULL, memory will be allocated internally
#     bool   no_alloc;   // don't allocate memory for the tensor data
# };
class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]

# struct ggml_context
ggml_context_p = NewType("ggml_context_p", int)
ggml_context_p_ctypes = ctypes.c_void_p


# struct ggml_context * ggml_init(struct ggml_init_params params);
@ctypes_function("ggml_init", [ggml_init_params], ggml_context_p_ctypes)
def ggml_init(params: ggml_init_params) -> ggml_context_p:
    ...


# void ggml_free(struct ggml_context * ctx)
@ctypes_function("ggml_free", [ggml_context_p_ctypes], None)
def ggml_free(ctx: ggml_context_p) -> None:
    ...