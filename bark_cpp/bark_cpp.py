import os
import pathlib
import ctypes
from typing import NewType

from bark_cpp._ctypes_extensions import load_shared_library, ctypes_function_for_shared_library


# Specify the base name of the shared library to load
_lib_base_name = "bark"
_override_base_path = os.environ.get("BARK_CPP_LIB_PATH")
_base_path = pathlib.Path(os.path.abspath(os.path.dirname(
    __file__))) / "lib" if _override_base_path is None else pathlib.Path(_override_base_path)
# Load the library
_lib = load_shared_library(_lib_base_name, _base_path)

ctypes_function = ctypes_function_for_shared_library(_lib)


# enum bark_verbosity_level {
#     LOW    = 0,
#     MEDIUM = 1,
#     HIGH   = 2,
# };
BARK_VERBOSITY_LEVEL_LOW = 0
BARK_VERBOSITY_LEVEL_MEDIUM = 1
BARK_VERBOSITY_LEVEL_HIGH = 2


# enum bark_encoding_step {
#     SEMANTIC = 0,
#     COARSE   = 1,
#     FINE     = 2,
# };
BARK_ENCODING_STEP_SEMANTIC = 0
BARK_ENCODING_STEP_COARSE = 1
BARK_ENCODING_STEP_FINE = 2


# enum ggml_ftype {
#     GGML_FTYPE_UNKNOWN     = -1,
#     GGML_FTYPE_ALL_F32     = 0,
#     GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
# };
GGML_FTYPE_UNKNOWN     = -1
GGML_FTYPE_ALL_F32     = 0
GGML_FTYPE_MOSTLY_F16  = 1 
GGML_FTYPE_MOSTLY_Q4_0 = 2 
GGML_FTYPE_MOSTLY_Q4_1 = 3  
GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
GGML_FTYPE_MOSTLY_Q8_0 = 7  
GGML_FTYPE_MOSTLY_Q5_0 = 8  
GGML_FTYPE_MOSTLY_Q5_1 = 9  
GGML_FTYPE_MOSTLY_Q2_K = 10 
GGML_FTYPE_MOSTLY_Q3_K = 11 
GGML_FTYPE_MOSTLY_Q4_K = 12 
GGML_FTYPE_MOSTLY_Q5_K = 13 
GGML_FTYPE_MOSTLY_Q6_K = 14 

# struct bark_context;
bark_context_p = NewType("bark_context_p", int)
bark_context_p_ctypes = ctypes.c_void_p

# typedef void (*bark_progress_callback)(struct bark_context * bctx, enum bark_encoding_step step, int progress, void * user_data);
bark_progress_callback = ctypes.CFUNCTYPE(
    None, bark_context_p_ctypes, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)


class bark_context_params(ctypes.Structure):
    _fields_ = [
        ("verbosity", ctypes.c_int),  # enum bark_verbosity_level
        ("temp", ctypes.c_float),
        ("fine_temp", ctypes.c_float),
        ("min_eos_p", ctypes.c_float),
        ("sliding_window_size", ctypes.c_int32),
        ("max_coarse_history", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("target_bandwidth", ctypes.c_int32),
        ("cls_token_id", ctypes.c_int32),
        ("sep_token_id", ctypes.c_int32),
        ("n_steps_text_encoder", ctypes.c_int32),
        ("text_pad_token", ctypes.c_int32),
        ("text_encoding_offset", ctypes.c_int32),
        ("semantic_rate_hz", ctypes.c_float),
        ("semantic_pad_token", ctypes.c_int32),
        ("semantic_vocab_size", ctypes.c_int32),
        ("semantic_infer_token", ctypes.c_int32),
        ("coarse_rate_hz", ctypes.c_float),
        ("coarse_infer_token", ctypes.c_int32),
        ("coarse_semantic_pad_token", ctypes.c_int32),
        ("n_coarse_codebooks", ctypes.c_int32),
        ("n_fine_codebooks", ctypes.c_int32),
        ("codebook_size", ctypes.c_int32),
        ("progress_callback", bark_progress_callback),
        ("progress_callback_user_data", ctypes.c_void_p),
    ]


# struct bark_context_params bark_context_default_params(void);
@ctypes_function("bark_context_default_params", [], bark_context_params)
def bark_context_default_params() -> bark_context_params:
    ...


# struct bark_context *bark_load_model(
#     const char *model_path,
#     struct bark_context_params params,
#     uint32_t seed);
@ctypes_function("bark_load_model", [ctypes.c_char_p, bark_context_params, ctypes.c_uint], bark_context_p_ctypes)
def bark_load_model(model_path: bytes, params: bark_context_params, seed: int) -> bark_context_p:
    ...


# bool bark_generate_audio(
#     struct bark_context *bctx,
#     const char *text,
#     int n_threads);
@ctypes_function("bark_generate_audio", [bark_context_p_ctypes, ctypes.c_char_p, ctypes.c_int], ctypes.c_bool)
def bark_generate_audio(bctx: bark_context_p, char: bytes, n_threads: int) -> int:
    ...


# int bark_get_audio_data_size(
#     struct bark_context *bctx);
@ctypes_function("bark_get_audio_data_size", [bark_context_p_ctypes], ctypes.c_int)
def bark_get_audio_data_size(bctx: bark_context_p) -> int:
    ...


# float *bark_get_audio_data(
#     struct bark_context *bctx);
@ctypes_function("bark_get_audio_data", [bark_context_p_ctypes], ctypes.POINTER(ctypes.c_float))
def bark_get_audio_data(bctx: bark_context_p) -> ctypes.Array[ctypes.c_float]:
    ...


# void bark_free(struct bark_context *bctx);
@ctypes_function("bark_free", [bark_context_p_ctypes], None)
def bark_free(bctx: bark_context_p) -> None:
    ...


# int64_t bark_get_load_time(struct bark_context *bctx);
@ctypes_function("bark_get_load_time", [bark_context_p_ctypes], ctypes.c_int64)
def bark_get_load_time(bctx: bark_context_p) -> int:
    ...


# int64_t bark_get_eval_time(struct bark_context *bctx);
@ctypes_function("bark_get_eval_time", [bark_context_p_ctypes], ctypes.c_int64)
def bark_get_eval_time(bctx: bark_context_p) -> int:
    ...


# bool bark_model_quantize(
#     const char *fname_inp,
#     const char *fname_out,
#     enum ggml_ftype ftype);
@ctypes_function("bark_model_quantize", [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int], ctypes.c_bool)
def bark_model_quantize(fname_inp: bytes, fname_out: bytes, ftype: int) -> bool:
    ...