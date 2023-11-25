from typing import Optional, Union

import torch
import transformers
from smoe.models.llama_moe import LlamaMoEForCausalLM

from lm_eval.base import BaseLM
from .gpt2 import _get_dtype
from .huggingface import _get_accelerate_args


class LlamaMoELM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device="cuda",
        pretrained="llama_moe",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        use_accelerate: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        device_list = set(
            ["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        )
        if device and device in device_list:
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        model_kwargs = {}
        if use_accelerate:
            model_kwargs = _get_accelerate_args(
                device_map_option,
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
            )
        self.model = LlamaMoEForCausalLM.from_pretrained(
            pretrained,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=low_cpu_mem_usage,
            revision=revision,
            torch_dtype=_get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        ).eval()
        self._device = device

        if not load_in_8bit:
            try:
                self.model.to(self.device)
            except Exception:
                print("Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore.")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        self.vocab_size = self.tokenizer.vocab_size

        # setup for automatic batch size detection
        if batch_size == "auto":
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size)

        self._max_length = max_length

        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate` can place `lm_head` weights on a different device than
            # the user specified one so we force `self._device` to be the same as
            # `lm_head`'s.
            self._device = self.model.hf_device_map["lm_head"]

        if not use_accelerate and not (load_in_4bit or load_in_8bit):
            try:
                self.model.to(self._device)
            except Exception:
                print("Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore.")

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)
