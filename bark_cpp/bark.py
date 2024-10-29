from contextlib import ExitStack, closing

import numpy.typing as npt
import numpy as np

from bark_cpp import bark_cpp
from scipy.io.wavfile import write as write_wav


class BarkContext:
    """
    Intermediate-level wrapper for a bark.cpp bark_context. 
    """

    def __init__(self,
                 model_path: bytes,
                 params: bark_cpp.bark_context_params,
                 seed: int = 0
                 ):
        self._exit_stack = ExitStack()
        bctx = bark_cpp.bark_load_model(model_path,
                                        params,
                                        seed)

        if bctx is None:
            raise RuntimeError("Failed to create bark context")
        self.ctx = bctx

        def free_ctx():
            if self.ctx is None:
                return
            bark_cpp.bark_free(self.ctx)
            self.ctx = None

        self._exit_stack.callback(free_ctx)

    def close(self):
        self._exit_stack.close()

    def __del__(self):
        self.close()

    @staticmethod
    def default_params() -> bark_cpp.bark_context_params:
        """
        Returns default parameters for a bark context.

        Returns:
            bark_cpp.bark_context_params: Default parameters for a bark context.
        """
        return bark_cpp.bark_context_default_params()

    def generate_audio(self, prompt: bytes, n_threads: int = 1) -> bool:
        """
        Generates an audio data from the given prompt.

        Args:
            prompt (bytes): The prompt to generate audio from.
            n_threads (int): The number of threads to use for generating the audio.

        Returns:
            bool: An integer indicating the success of the audio generation process.
        """
        return bark_cpp.bark_generate_audio(self.ctx, prompt, n_threads)

    def get_audio_data(self):
        """
        Gets the audio data from the bark context.

        Returns:
            ctypes.Array[c_float]: The audio data.
        """
        return bark_cpp.bark_get_audio_data(self.ctx)

    def get_audio_data_size(self):
        """
        Gets the size of the audio data from the bark context.

        Returns:
            int64: The size of the audio data.
        """
        return bark_cpp.bark_get_audio_data_size(self.ctx)

    def get_load_time(self):
        """
        Gets the load time of the model in microseconds.

        Returns:
            int64: The load time in microseconds.
        """
        return bark_cpp.bark_get_load_time(self.ctx)

    def get_eval_time(self):
        """
        Gets the evaluation time of last audio generation in microseconds.

        Returns:
            int64: The evaluation time in microseconds.
        """
        return bark_cpp.bark_get_eval_time(self.ctx)


class Bark:
    """
    High-level Python wrapper for a Bark model.
    """

    def __init__(
            self,
            model_path: str,
            temp: float = 0.7,
            fine_temp: float = 0.5,
            min_eos_p: float = 0.2,
            sliding_window_size: int = 60,
            max_coarse_history: int = 630,
            sample_rate: int = 24000,
            target_bandwidth: int = 6,
            n_steps_text_encoder: int = 768,
            semantic_rate_hz: float = 49.9,
            coarse_rate_hz: float = 75.,
            seed: int = 0):
        self._stack = ExitStack()
        self.model_path = model_path

        # Context params
        self.ctx_params = bark_cpp.bark_context_default_params()
        self.ctx_params.temp = temp
        self.ctx_params.fine_temp = fine_temp
        self.ctx_params.min_eos_p = min_eos_p
        self.ctx_params.sliding_window_size = sliding_window_size
        self.ctx_params.max_coarse_history = max_coarse_history
        self.ctx_params.sample_rate = sample_rate
        self.ctx_params.target_bandwidth = target_bandwidth
        self.ctx_params.n_steps_text_encoder = n_steps_text_encoder
        self.ctx_params.semantic_rate_hz = semantic_rate_hz
        self.ctx_params.coarse_rate_hz = coarse_rate_hz

        self._ctx = self._stack.enter_context(
            closing(
                BarkContext(
                    model_path=model_path.encode(),
                    params=self.ctx_params,
                    seed=seed
                )
            )
        )

    def close(self):
        self._stack.close()

    def generate_audio(self, prompt: str, n_threads: int = 1) -> npt.NDArray[np.float32]:
        assert self._ctx is not None

        is_success = self._ctx.generate_audio(
            prompt.encode('utf-8'), n_threads)
        if not is_success:
            raise RuntimeError("Failed to generate audio")
        array_size = self._ctx.get_audio_data_size()
        audio_ptr = self._ctx.get_audio_data()
        audio_arr = np.ctypeslib.as_array(audio_ptr, shape=(array_size,))
        return audio_arr

    def get_load_time(self) -> int:
        assert self._ctx is not None
        return self._ctx.get_load_time()

    def get_eval_time(self) -> int:
        assert self._ctx is not None
        return self._ctx.get_eval_time()

    @staticmethod
    def write_wav(filename: str, audio_arr: npt.NDArray[np.float32]):
        SAMPLE_RATE = 24000
        write_wav(filename, SAMPLE_RATE, audio_arr)

    @staticmethod
    def quantize_model(in_model_fname: str, out_model_fname: str, ftype: int) -> bool:
        return bark_cpp.bark_model_quantize(in_model_fname.encode(), out_model_fname.encode(), ftype)
