'''
Outline:

Have attacker generate a bunch of queries using custom prompt <- can use existing prompts
Have victim generate responses
Filter for correctly safe queries
'''

from ..redteaming import Redteam, Agent, Judge, suites
from ..qwen_llm import QwenLLM
from vllm.sampling_params import SamplingParams

sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=24576,
)
redteam_llm = QwenLLM(
    "Qwen/Qwen2.5-7B-Instruct", vllm_args={"vllm_gpu_memory_utilization": 0.45, "vllm_max_model_len": 20544}, sampling_params=sampling_params
)

redteam = Redteam(
    redteam_llm, system_message=suite.system_description, examples=examples
)
