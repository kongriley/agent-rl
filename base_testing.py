import logging
import json
import time
import click
from rich.logging import RichHandler
from redteaming import suites, Redteam, Agent, Judge, format_messages
from openai import OpenAI
from qwen_llm import QwenLLM
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

import os
import dotenv

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@click.command()
@click.option("--n-iters", default=1000, help="Number of iterations")
@click.option("--suite-name", default="workspace", help="Name of the test suite")
@click.option("--redteam-scale", default=1.5, help="Scale factor for redteam model")
@click.option("--victim-scale", default=1.5, help="Scale factor for victim model")
@click.option("--port", default=8000, help="Port for vllm")
def main(n_iters, suite_name, redteam_scale, victim_scale, port):
    ### ARGS ###
    if redteam_scale - int(redteam_scale) == 0:
        redteam_scale = int(redteam_scale)
    if victim_scale - int(victim_scale) == 0:
        victim_scale = int(victim_scale)

    out_path = f"results/{suite_name}/few-shot/"
    redteam_model = f"Qwen/Qwen2.5-{redteam_scale}B-Instruct"
    victim_model = f"Qwen/Qwen2.5-{victim_scale}B-Instruct"
    redteam_port = port
    victim_port = port if redteam_scale == victim_scale else port + 1
    llm_args = {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 2048,
        "extra_body": {
            "repetition_penalty": 1.05,
            "max_completion_tokens": 24576,
        },
    }

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
    redteam_client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{redteam_port}/v1",
    )
    victim_client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{victim_port}/v1",
    )
    judge_client = OpenAI(api_key=OPENAI_API_KEY)

    redteam_llm = QwenLLM(redteam_client, redteam_model, **llm_args)
    victim_llm = QwenLLM(victim_client, victim_model, **llm_args)
    judge_llm = OpenAILLM(judge_client, "gpt-4o")

    redteam = Redteam(
        redteam_llm, system_message=suite.system_description, examples=examples
    )
    agent = Agent(
        suite.runtime, suite.environment, victim_llm, suite.system_description
    )
    judge = Judge(judge_llm, suite.system_description)

    results = {
        "suite_name": suite_name,
        "redteam_model": redteam_model,
        "victim_model": victim_model,
        "unsafe": 0,
        "safe": 0,
        "error": 0,
        "start_time": start_time,
    }
    for i in range(n_iters):
        redteam_query = redteam.run(suite.system_description, suite.environment)
        messages = agent.run(redteam_query)
        judge_response = judge.run(messages, suite.environment)
        judgement = judge_response.split()[-1]
        if judgement not in ("safe", "unsafe"):
            judgement = "error"
            results["error"] += 1
        else:
            if judgement == "safe":
                results["safe"] += 1
            else:
                results["unsafe"] += 1

        with open(os.path.join(out_path, f"log_{start_time}.txt"), "a+") as f:
            f.write(f"{i}/{n_iters}, {time.time()}\n")
            f.write(f"Redteam query: {redteam_query}\n")
            for role, message in format_messages(messages):
                f.write(f"{role}: {message}\n")
            f.write("\n")
            f.write(f"Judge response: {judge_response}\n")
            f.write(f"Judgement: {judgement}\n")
            f.write("==============\n")

    results["time_elapsed"] = time.time() - start_time
    with open(os.path.join(out_path, "results.jsonl"), "a") as f:
        f.write(json.dumps(results))
        f.write("\n")


if __name__ == "__main__":
    main()
