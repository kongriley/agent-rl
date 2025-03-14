import logging
import json
import time
import click
from rich.logging import RichHandler
from redteaming import suites, Redteam, Agent, Judge, format_messages, refusal_prompt
from openai import OpenAI
from qwen_llm import QwenLLM
from vllm import SamplingParams
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from utils.self_bleu import SelfBleuReward

import os
import dotenv

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@click.command()
@click.option("--n-iters", default=500, help="Number of iterations")
@click.option("--suite-name", default="workspace", help="Name of the test suite")
@click.option("--mode", default="zero-shot", help="Mode of testing (few-shot/zero-shot)")
@click.option("--redteam-scale", default=1.5, help="Scale factor for redteam model")
@click.option("--log-results", default=True, help="Whether to log the results")
def main(n_iters, suite_name, mode, redteam_scale, log_results):
    redteam_scale = int(redteam_scale) if redteam_scale.is_integer() else redteam_scale

    out_path = f"results/{suite_name}/{mode}/"
    redteam_model = f"Qwen/Qwen2.5-{redteam_scale}B-Instruct"
    victim_model = "gpt-4o-mini"
    redteam_vllm_args = {
        "vllm_gpu_memory_utilization": 0.45,
        "vllm_max_model_len": 20544,
    }
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=24576,
    )

    assert out_path.endswith("/"), "out_path must end in /"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )

    start_time = time.time()

    ### SETUP ###
    suite = suites[suite_name]
    examples = suite.examples

    # Initialize models
    judge_client = OpenAI(api_key=OPENAI_API_KEY)

    redteam_llm = QwenLLM(
        redteam_model, vllm_args=redteam_vllm_args, sampling_params=sampling_params
    )
    victim_llm = OpenAILLM(judge_client, victim_model)
    judge_llm = OpenAILLM(judge_client, "gpt-4o")

    redteam = Redteam(
        redteam_llm, system_message=suite.system_description
    )
    if mode == "few-shot":
        assert examples is not None, "Few-shot mode requires examples"
        redteam.examples = examples
        
    agent = Agent(
        suite.runtime, suite.environment, victim_llm, suite.system_description
    )
    judge = Judge(judge_llm, suite.system_description)
    refusal_judge = Judge(judge_llm, suite.system_description)

    results = {
        "suite_name": suite_name,
        "redteam_model": redteam_model,
        "victim_model": victim_model,
        "unsafe": {
            "refusal": 0,
            "no_refusal": 0,
        },
        "safe": {
            "refusal": 0,
            "no_refusal": 0,
        },
        "error": 0,
        "start_time": start_time,
    }
    bleu_module = SelfBleuReward()

    queries = []
    for i in range(n_iters):
        redteam_query = redteam.run(suite.system_description, suite.environment)
        queries.append(redteam_query)
        bleu_module.append_reference(redteam_query)

        messages = agent.run(redteam_query)
        judge_response = judge.run(messages)
        judgement = judge_response.split()[-1]
        if judgement not in ("safe", "unsafe"):
            judgement = "error"
            results["error"] += 1
        else:
            result_dict = results["safe"] if judgement == "safe" else results["unsafe"]

            refusal_response = refusal_judge.run(messages, custom_prompt=refusal_prompt)
            refusal_judgement = refusal_response.split()[-1]
            if refusal_judgement == "yes":
                result_dict["refusal"] += 1
            elif refusal_judgement == "no":
                result_dict["no_refusal"] += 1
            else:
                refusal_judgement = "error"
                results["error"] += 1

        if log_results:
            with open(os.path.join(out_path, f"log_{start_time}.txt"), "a+") as f:
                f.write(f"{i}/{n_iters}, {time.time()}\n")
                f.write(f"Redteam query: {redteam_query}\n")
                for role, message in format_messages(messages):
                    f.write(f"{role}: {message}\n")
                f.write(f"Judge response: {judge_response}\n")
                f.write(f"Judgement: {judgement}\n")
                f.write(f"Refusal response: {refusal_response}\n")
                f.write(f"Refusal judgement: {refusal_judgement}\n")
                f.write("==============\n")

    # TODO: Fix BLEU score calculation (ffi_prep_cif_var failed)
    # results["bleu_score"] = bleu_module(queries)

    results["time_elapsed"] = time.time() - start_time
    with open(os.path.join(out_path, "results.jsonl"), "a") as f:
        f.write(json.dumps(results))
        f.write("\n")


if __name__ == "__main__":
    main()
