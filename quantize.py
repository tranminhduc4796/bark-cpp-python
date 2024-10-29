"""
Migration script from [examples/quantize](https://github.com/PABannier/bark.cpp/blob/main/examples/quantize/main.cpp)
Quantize a Bark model from f16 to a quantized format.
"""
import argparse
import ctypes

from bark_cpp.bark_cpp import GGML_FTYPE_MOSTLY_Q4_0, GGML_FTYPE_MOSTLY_Q4_1, GGML_FTYPE_MOSTLY_Q5_0, GGML_FTYPE_MOSTLY_Q5_1, GGML_FTYPE_MOSTLY_Q8_0
from bark_cpp import Bark
from bark_cpp._ggml import ggml_init, ggml_free, ggml_init_params


FMT_STR_TO_FTYPE = {
    "q4_0": GGML_FTYPE_MOSTLY_Q4_0,
    "q4_1": GGML_FTYPE_MOSTLY_Q4_1,
    "q5_0": GGML_FTYPE_MOSTLY_Q5_0,
    "q5_1": GGML_FTYPE_MOSTLY_Q5_1,
    "q8_0": GGML_FTYPE_MOSTLY_Q8_0
}


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a Bark model from f32 to a quantized format.")
    parser.add_argument("input_model", type=str,
                        help="Path to the input model file in f32 format.")
    parser.add_argument("output_model", type=str,
                        help="Path to save the quantized model.")
    parser.add_argument("quantization_type", type=str, 
                        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                        help="Quantization type.")

    args = parser.parse_args()

    # Initialize ggml context (this might be needed before using Bark)
    params = ggml_init_params(
        mem_size=ctypes.c_size_t(0),
        mem_buffer=ctypes.c_void_p(None),
        no_alloc=ctypes.c_bool(False)
    )
    ctx = ggml_init(params)
    ggml_free(ctx)

    # Perform model quantization
    Bark.quantize_model(args.input_model, args.output_model,
                        FMT_STR_TO_FTYPE[args.quantization_type])

    print(f"Model quantized and saved to: {args.output_model}")


if __name__ == "__main__":
    main()
