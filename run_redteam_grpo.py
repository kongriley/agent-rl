#!/usr/bin/env python3
"""
Script to run the GRPO trainer for red-teaming LLMs.
"""

import argparse
import os
from typing import List

import torch
import yaml
from accelerate import Accelerator

from trlx.data.default_configs import TRLConfig
from trlx.utils import set_seed
from trlx.utils.loading import get_trainer

from accelerate_redteam_grpo_trainer import AccelerateRedteamGRPOTrainer, RedteamGRPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training for red-teaming LLMs")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--reward_fn", type=str, required=True, help="Path to reward function module")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo_redteam", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config from YAML file
    with open(args.config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Initialize config
    config = TRLConfig.from_dict(config_dict)
    
    # Set output directory in config
    config.train.project_name = os.path.basename(args.output_dir)
    config.train.run_name = "grpo_redteam"
    config.train.logging_dir = args.output_dir
    
    # Import reward function dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward_module", args.reward_fn)
    reward_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_module)
    
    # Initialize reward function
    reward_fn = reward_module.get_reward_fn()
    
    # Initialize trainer
    trainer = get_trainer(config.train.trainer)(
        config=config,
        reward_fn=reward_fn,
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main() 