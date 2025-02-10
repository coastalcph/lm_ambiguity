from collections import deque
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Generator, List, Type

from outlines.serve.vllm import JSONLogitsProcessor
from pydantic import BaseModel
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams

from .dataset import TsvTextDataset
from .utils import parse_json, parse_pydantic_schema, validate_json_with_schema


@dataclass
class JsonCompletion:
    text: str
    output_json: Dict[str, Any]

    def values_as_list(self) -> List[Any]:
        return [self.text] + list(self.output_json.values())

    def keys_as_list(self) -> List[str]:
        return ["text"] + list(self.output_json.keys())
        
    
class Engine:
    def __init__(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
        schema: Type[BaseModel],
        dataset: TsvTextDataset,
        enable_logprobs: bool = False,  # New parameter to enable/disable logprobs
    ):
        self.llm = llm
        self.sampling_params = sampling_params
        self.schema = schema
        self.dataset = dataset
        self.enable_logprobs = enable_logprobs

        self.sampling_params.logits_processors = [
            JSONLogitsProcessor(schema=self.schema, llm=self.llm, whitespace_pattern=r" ?")
        ]

    @cached_property
    def schema_string(self) -> str:
        return parse_pydantic_schema(self.schema)
    
    def process_batch(self, batch: List[str]) -> List[JsonCompletion]:
        json_completions = [JsonCompletion(text=text, output_json={}) for text in batch]
        batch_log_probs = [None] * len(batch) if self.enable_logprobs else None
        batch_log_ambiguous = [None] * len(batch)  # To store log_ambiguous predictions

        queue = deque([(i, text) for i, text in enumerate(batch)])

        while queue:
            indices = []
            prompts = []
            while queue:
                i, text = queue.popleft()
                indices.append(i)
                prompts.append(self.dataset.template.render(text=text, schema=self.schema_string))

            raw_outputs = self.llm.generate(
                prompts, sampling_params=self.sampling_params, use_tqdm=True
            )

            for i, raw_output in zip(indices, raw_outputs):
                json_output = parse_json(raw_output.outputs[0].text)
                log_probs = raw_output.outputs[0].promptlogprobs if self.enable_logprobs else None
                
                if not validate_json_with_schema(json_output, self.schema):
                    queue.append((i, batch[i]))
                else:
                    json_completions[i].output_json = json_output
                    if self.enable_logprobs:
                        batch_log_probs[i] = log_probs

        if self.enable_logprobs:
            return json_completions, batch_log_probs
        else:
            return json_completions
        
    def __call__(self) -> Generator:
        for batch in tqdm(self.dataset, desc="Processing batches"):
            result = self.process_batch(batch)
            if self.enable_logprobs:
                json_completions, batch_log_probs = result
                for json_completion, log_probs in zip(json_completions, batch_log_probs):
                    yield json_completion, log_probs
                    
            else:
                json_completions, batch_log_ambiguous = result
                for json_completion, log_ambiguous in zip(json_completions, batch_log_ambiguous):
                    yield json_completion, log_ambiguous
