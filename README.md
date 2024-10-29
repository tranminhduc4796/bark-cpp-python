<p align="center">
  <img src="https://github.com/user-attachments/assets/aa10d9b8-1539-40da-a974-6d2dab6a1809" style="max-width: 100%; height: auto;"/>
</p>

# 🐶 bark-cpp-python 🐍

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
![Python](https://img.shields.io/badge/python-3.10%2B-pink.svg)


Python bindings for [bark.cpp](https://github.com/PABannier/bark.cpp) using `ctypes`. Utilize the power of GGML with bark, one of the most popular TTS models, and its quantized versions through a friendly Python interface 🔥🔥🔥.

## ⚙️ Feature
Inpsired by [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), this package provides:

* [x] Low-level access to C API via `ctypes` interface
* [x] High-level Python API for TTS

## 🚀 Demo
This demo is on `AMD Ryzen 5 5600H`, `Ubuntu 20.04`
```bash
$ python demo.py ./models/bark-small/ggml_weights_q4_1.bin -p "Hi, I am Bark. Nice to meet you" -t 8 --dest output.wav

                 ___       _      ___     __  ___
 /\__/\  woof   |    \    / \    |    \  |  |/  /
/      \  woof  |    /   /   \   |    /  |     /
\      /        |    \  /  _  \  |  _ \  |     \
 \____/         |____/ /__/ \__\ |_| |_\ |__|\__\
    

encodec_load_model_weights: in_channels = 1
encodec_load_model_weights: hidden_dim  = 128
encodec_load_model_weights: n_filters   = 32
encodec_load_model_weights: kernel_size = 7
encodec_load_model_weights: res_kernel  = 3
encodec_load_model_weights: n_bins      = 1024
encodec_load_model_weights: bandwidth   = 24
encodec_load_model_weights: sample_rate = 24000
encodec_load_model_weights: ftype       = 1
encodec_load_model_weights: qntvr       = 0
encodec_load_model_weights: ggml tensor size    = 320 bytes
encodec_load_model_weights: backend buffer size =  54.36 MB
encodec_load_model_weights: using CPU backend
encodec_load_model_weights: model size =    44.36 MB
encodec_load_model: n_q = 32

bark_tokenize_input: prompt: 'Hi, I am Bark. Nice to meet you'
bark_tokenize_input: number of tokens in prompt = 513, first 8 tokens: 30113 10165 10194 20440 30746 20222 10167 36966 



bark_print_statistics:   sample time =    49.21 ms / 455 tokens
bark_print_statistics:  predict time =  3471.03 ms / 7.63 ms per token
bark_print_statistics:    total time =  3542.42 ms



bark_print_statistics:   sample time =    21.86 ms / 1364 tokens
bark_print_statistics:  predict time = 33798.57 ms / 24.78 ms per token
bark_print_statistics:    total time = 33829.69 ms



bark_print_statistics:   sample time =    70.14 ms / 6144 tokens
bark_print_statistics:  predict time =  8684.00 ms / 1.41 ms per token
bark_print_statistics:    total time =  8783.56 ms

encodec_eval: compute buffer size: 230.30 MB

Evaluated time: 47.49s
```
[output.webm](https://github.com/user-attachments/assets/5e1ca97c-f81f-4bc2-8118-41b007e7c33e)


## 🔧 Installation

### Pip
```bash
pip install bark-cpp-python
```

### Build from source

1. Clone the repo and submodules
```bash
git clone --recursive https://github.com/tranminhduc4796/bark-cpp-python.git

cd bark-cpp-python
```
2. Build and install
```bash
pip install .
```

<details>
<summary>🤖 Debug</summary>

#### GLIBCXX_3.4.32 not found
If you meet this error when `import bark_cpp`:
```bash
RuntimeError: Failed to load shared library '~/miniconda3/envs/bark_cpp/lib/python3.10/site-packages/bark_cpp/lib/libbark.so': ~/miniconda3/envs/bark_cpp/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by ~/miniconda3/envs/bark_cpp/lib/python3.10/site-packages/bark_cpp/lib/libencodec.so)

```
Install the latest gcc with:
```bash
conda install -c conda-forge gcc
```
</details>


## 🐕 Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Download the Bark checkpoints and vocabulary
python3 download_weights.py --out-dir ./models --models bark-small bark

# Convert the model to ggml format
python3 convert.py --dir-model ./models/bark-small --use-f16

# Quantize model (Optional), must enable --use-f16 in the above command
python quantize.py ./models/bark-small/ggml_weights.bin ./models/bark-small/ggml_weights_q4_1.bin q4_1

# Run the demo
python demo.py ./models/bark-small/ggml_weights.bin -p "Hi, I am Bark. Nice to meet you" -t 8 --dest output.wav
```


## 🐍 High-level Python API
```python
args = parse_arguments()

bark = Bark(
        model_path=args.model_path,
        temp=args.temp,
        fine_temp=args.fine_temp,
        min_eos_p=args.min_eos_p,
        sliding_window_size=args.sliding_window_size,
        max_coarse_history=args.max_coarse_history,
        sample_rate=args.sample_rate,
        target_bandwidth=args.target_bandwidth,
        n_steps_text_encoder=args.n_steps_text_encoder,
        semantic_rate_hz=args.semantic_rate_hz,
        coarse_rate_hz=args.coarse_rate_hz,
        seed=args.seed
    )
audio_arr = bark.generate_audio(args.prompt, args.threads)

print("Evaluated time: {:.2f}s".format(bark.get_eval_time() / 1e6))
bark.write_wav(args.dest, audio_arr)
```

## Acknowledgments
* [Suno AI's bark](https://github.com/suno-ai/bark)
* [bark.cpp](https://github.com/PABannier/bark.cpp)
