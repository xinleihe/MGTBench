import os
import sys
import torch
import hashlib
from itertools import chain
from typing import List, Literal, Optional, Tuple
import logging
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def load_pretrained(model_name_or_path, quantization_bit=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    config_kwargs = {
        "trust_remote_code": True,
        # "_attn_implementation": 'flash_attention_2'
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        padding_side="left",
        **config_kwargs
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
        tokenizer.pad_token_id = 0 # set as the <unk> token

    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)

    # Quantization configurations (using bitsandbytes library).
    if quantization_bit is not None:
        if quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("transformers>=4.30.1", "To fix: pip install transformers>=4.30.1")
            require_version("accelerate>=0.20.3", "To fix: pip install accelerate>=0.20.3")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4"
            )
        is_mergeable = False
        # config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        logger.info("Quantizing model to {} bit.".format(quantization_bit))

    config_kwargs["device_map"] = "auto"


    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype= torch.float16,
        **config_kwargs
    )
    return model, tokenizer


def load_pretrained_mask(model_name_or_path, quantization_bit=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    config_kwargs = {
        "trust_remote_code": True,
    }

    
    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    
    # Quantization configurations (using bitsandbytes library).
    if quantization_bit is not None:
        if quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("transformers>=4.30.1", "To fix: pip install transformers>=4.30.1")
            require_version("accelerate>=0.20.3", "To fix: pip install accelerate>=0.20.3")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4"
            )
        is_mergeable = False
        # config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        logger.info("Quantizing model to {} bit.".format(quantization_bit))

    config_kwargs["device_map"] = "auto"
    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype= torch.float16,
        **config_kwargs
    )
    try:
        n_positions = model.config.n_positions
    except AttributeError:
        n_positions = 512

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        model_max_length=n_positions,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
        tokenizer.pad_token_id = 0 # set as the <unk> token

    return model, tokenizer

def load_pretrained_supervise(model_name_or_path, kargs, quantization_bit=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    config_kwargs = {
        "trust_remote_code": True,
    } 
    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    if 'num_labels' in kargs:
        config.num_labels = kargs['num_labels']
    config_kwargs["device_map"] = "cuda"


    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype= torch.float16,
        ignore_mismatched_sizes=True,
        **config_kwargs
    )
    try:
        n_positions = model.config.n_positions
    except AttributeError:
        n_positions = 512
    tokenizer_path = kargs.get("tokenizer_path", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        model_max_length=n_positions,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
        tokenizer.pad_token_id = 0 # set as the <unk> token

    return model, tokenizer

