from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, II


@dataclass
class ModelConfig:
    _target_: str = "vllm.LLM"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int = 2488
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    max_logprobs: int = 20


@dataclass
class GenerationConfig:
    _target_: str = "vllm.SamplingParams"
    max_tokens: int = 200
    temperature: float = 0.4
    top_k: int = 50
    top_p: float = 1.0
    logprobs: int = II("generation.top_k")
    prompt_logprobs: int = II("generation.top_k")


@dataclass
class DataConfig:
    _target_: str = "llm_inference.dataset.TsvTextDataset"
    tsv_path: str = MISSING
    template_name: str = "llama3"
    text_column: str = "sentence"
    batch_size: int = 1


@dataclass
class RunConfig:
    output_path: str = MISSING
    model: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    data: DataConfig = DataConfig()
    schema_name: str = "ambiguity_check"


cs = ConfigStore.instance()
cs.store(name="config", node=RunConfig)

