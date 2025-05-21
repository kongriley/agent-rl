import contextlib
import inspect
import json
import os
import sys
import functools
from abc import abstractmethod
from contextlib import contextmanager
from time import time
from typing import Dict, List, Optional, Tuple, Any

from dataclasses import dataclass, field
from torchtyping import TensorType

import yaml
import ray
import torch
from accelerate import Accelerator  # type: ignore
from ray.air import session
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

import torch.nn as nn
import torch.nn.functional as F

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.pipeline import MiniBatchIterator
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.utils import (
    filter_non_scalars,
    get_distributed_config,
    get_git_tag,
    get_optimizer_class,
    get_scheduler_class,
    significant,
)
from trlx.utils.modeling import (
    flatten_dict,
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    gather_dict,
)

import nltk

import json
import os
import uuid
from time import time
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import Clock, infinite_dataloader
from trlx.utils.modeling import RunningMoments, gather_dict, logprobs_of_labels

from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.models.modeling_ppo import PPOConfig

from trlx.data.method_configs import MethodConfig, register_method

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

import pandas as pd

from self_bleu import SelfBleuReward
from sentence_embed import CosineSentenceEmbeddingReward

from clean_reward import GiberishPenalty

from PIL.Image import Image

logger = logging.get_logger(__name__)

class TextCSVLogger(object):

    def __init__(self, log_dir, output_filename):
        self.log_dir = log_dir
        self.output_filename = output_filename
        self.base_filename = os.path.splitext(output_filename)[0]
        self.iter_count = 0

    def _make_image_paths(self, n):
        base_dir = os.path.join(self.log_dir, self.base_filename, "images", f"{self.iter_count:05d}")
        os.makedirs(base_dir, exist_ok=True)
        return [f"{base_dir}/{i:03d}.png" for i in range(n)]

    def log(self, attacker_prompts, attacker_outputs, victim_outputs, scores):
        if isinstance(victim_outputs[0], Image):
            image_paths = self._make_image_paths(len(victim_outputs))
            for image_path, image in zip(image_paths, victim_outputs):
                image.save(image_path)
            victim_outputs = image_paths

        str_df = pd.DataFrame({
            "attacker's prompts": attacker_prompts,
            "attacker's responses": attacker_outputs,
            "victim's responses": victim_outputs,
            "score": scores,
            "iter": self.iter_count,
        })        
        str_df.to_csv(
            os.path.join(self.log_dir, self.output_filename),
            mode='w' if (self.iter_count == 0) else 'a', header=(self.iter_count == 0),
            sep="\t")
        self.iter_count += 1

class TextCSVLoggerWithTimestamp(object):

    def __init__(self, log_dir, output_filename):
        self.log_dir = log_dir
        self.output_filename = output_filename
        self.iter_count = 0 

    def log(self, attacker_prompts, attacker_outputs, victim_outputs, scores, timestamp):
        str_df = pd.DataFrame({
            "attacker's prompts": attacker_prompts,
            "attacker's responses": attacker_outputs,
            "victim's responses": victim_outputs,
            "score": scores,
            "iter": self.iter_count,
            "timestamp": timestamp,
        })        
        str_df.to_csv(
            os.path.join(self.log_dir, self.output_filename),
            mode='w' if (self.iter_count == 0) else 'a', header=(self.iter_count == 0),
            sep="\t")
        self.iter_count += 1

@dataclass
@register_method
class RedteamGRPOConfig(PPOConfig):
    """
    GRPO specific configuration
    """
    num_generations: int = 4        # Number of generations per prompt
    temperature: float = 1.0        # Temperature for sampling
    top_p: float = 1.0              # Top-p sampling parameter
    loss_type: str = "kl"           # Type of loss to use ("kl" or "pg")
    
    """
    BLEU rewards configuration
    """
    bleu_reward_coef: float = -0.5  # NOTE: must be negative since we want to minimize overlap
    bleu_reward_grams: str = "[3, 4, 5]"  # NOTE: accelerate tracker cannot log list arguments
    bleu_reward_include_prompts: bool = False  # Including prompts in continuation tasks
    bleu_tokenizer: str = "nltk"
    bleu_n_samples: int = -1

    """
    Entropy bonus configuration (i.e., KL penalty to uniform distribution)
    """
    ent_reward_coef: float = 0.0

    """
    Sentence embedding bonus
    """
    cossimemb_reward_coef: float = 0.0
    cossimemb_n_samples: int = -1
    cossimemb_impl: str = "huggingface"
    cossimemb_reward_include_prompts: bool = True
    cossimemb_model_device: str = "default" 
    
    """
    Textual similarity reward (between attacker's prompts and attacker's responses)
    """
    textual_sim_reward_coef: float = 0.0
    textual_sim_reward_include_prompts: bool = False
    
    """
    Target model's batch embedding diversity
    """
    target_sim_div_reward_coef: float = 0.0
    
    """
    GiberishPenalty
    """
    giberish_penalty_coef: float = 0.0
    giberish_model_device: str = "default"  # same as attacker
    
    """
    Reward model device
    """
    reward_model_device_offset: int = 0


@register_trainer
class AccelerateRedteamGRPOTrainer(AccelerateRLTrainer):
    """
    GRPO (Gradient-based Reinforcement Learning with Policy Optimization) trainer for red-teaming.
    
    GRPO differs from PPO in that it:
    1. Uses direct policy gradients rather than clipped surrogate objectives
    2. Generates multiple samples per prompt to estimate policy gradients
    3. Uses a different loss function based on KL-divergence or policy gradients
    """

    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._setup_grpo(config)

    def _setup_grpo(self, config):
        if inspect.isclass(self.reward_fn) or isinstance(self.reward_fn, functools.partial):
            self.reward_fn = self.reward_fn(self.accelerator.device, self.model.base_model, self.tokenizer)
        
        if self.config.method.bleu_tokenizer == "nltk":
            bleu_tokenizer = nltk.word_tokenize
            print(f"BLEU tokenizer: {bleu_tokenizer}")
        elif self.config.method.bleu_tokenizer == "model":
            print(f"BLEU tokenizer: {self.tokenizer}")
            bleu_tokenizer = lambda x: self.tokenizer.batch_decode(self.tokenizer(x, return_tensors="pt")["input_ids"][0].unsqueeze(1))
        
        self.bleu_reward_module = SelfBleuReward(
            grams=eval(config.method.bleu_reward_grams),
            sample_size=config.method.bleu_n_samples,
            tokenizer=bleu_tokenizer,
        )
        
        self.cossimemb_reward_module = CosineSentenceEmbeddingReward(
            n_samples=config.method.cossimemb_n_samples,
            impl=config.method.cossimemb_impl,
            device=(self.accelerator.device if config.method.cossimemb_model_device == "default" else config.method.cossimemb_model_device)
        )
        
        if self.config.method.giberish_penalty_coef != 0:
            self.giberish_penalty_penalty_module = GiberishPenalty(
                (self.accelerator.device if config.method.giberish_model_device == "default" else config.method.giberish_model_device)
            )
    
        self.train_text_logger = TextCSVLogger(self.accelerator.project_dir, "train.csv")
        self.eval_text_logger = TextCSVLogger(self.accelerator.project_dir, "eval.csv")
        self.history_scores = []
        
        # GRPO specific parameters
        self.num_generations = self.config.method.num_generations
        self.running_moments = RunningMoments()
        self.ref_mean = None
        self.ref_std = None

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """
        Generate samples from the model using the given inputs.
        For GRPO, we generate multiple samples per input.
        """
        # Set generation parameters
        gen_kwargs = dict(
            max_new_tokens=self.config.train.gen_kwargs.max_new_tokens,
            temperature=self.config.method.temperature,
            top_p=self.config.method.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_kwargs.update(kwargs)
        
        # Prepare inputs for generation
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # For each input, generate multiple samples
        all_samples = []
        for i in range(batch_size):
            # Repeat the input num_generations times
            repeated_input = input_ids[i:i+1].repeat(self.num_generations, 1)
            repeated_mask = None
            if attention_mask is not None:
                repeated_mask = attention_mask[i:i+1].repeat(self.num_generations, 1)
            
            # Generate samples
            with torch.no_grad():
                samples = self.model.generate(
                    input_ids=repeated_input,
                    attention_mask=repeated_mask,
                    **gen_kwargs
                )
                all_samples.append(samples)
        
        # Concatenate all samples
        return torch.cat(all_samples, dim=0)

    def _aggregate_traj_reward(self, all_scores, all_bleu_scores, all_cossimemb_scores, all_textualsim_scores, all_target_sim_div_scores, all_giberish_scores, device):
        """Aggregate trajectory rewards from different reward components"""
        return [
            torch.tensor(score + 
                self.config.method.bleu_reward_coef * bleu_score +
                self.config.method.cossimemb_reward_coef * cossimemb_score +
                self.config.method.textual_sim_reward_coef * textualsim_score + 
                self.config.method.target_sim_div_reward_coef * target_sim_div_score +
                self.config.method.giberish_penalty_coef * giberish_score
                , dtype=torch.float, device=device).view(
                -1,
            )
            for score, bleu_score, cossimemb_score, textualsim_score, target_sim_div_score, giberish_score in zip(
                all_scores, all_bleu_scores, all_cossimemb_scores, all_textualsim_scores, all_target_sim_div_scores, all_giberish_scores)
        ]

    @torch.inference_mode()
    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Generate experiences using GRPO approach.
        
        In GRPO, we:
        1. Generate multiple samples for each prompt
        2. Compute rewards for each sample
        3. Optimize directly using policy gradients instead of PPO's clipped surrogate objectives
        
        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            position=logging.get_verbosity() >= logging.WARNING,
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        grpo_batch_data = []
        accumulated_stats = []
        
        # Generate samples until we have enough rollouts
        samples_collected = 0
        while samples_collected < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model
            attention_mask_arg = batch["attention_mask"] if batch["attention_mask"].shape[0] != 1 else None
            samples = self.generate(batch["input_ids"], attention_mask_arg)
            stats["time/rollout_generate"] = time() - rollout_generate_time

            prompt_tensors = batch.input_ids
            device = samples.device
            
            # Number of unique prompts (each prompt has multiple generations)
            num_prompts = prompt_tensors.shape[0]
            
            # Prepare the data for gathering across processes
            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors) * self.num_generations, device=device)
            
            # Repeat prompts to match the number of samples generated
            repeated_prompts = []
            for i in range(num_prompts):
                repeated_prompts.append(prompt_tensors[i:i+1].repeat(self.num_generations, 1))
            repeated_prompts = torch.cat(repeated_prompts, dim=0)
            
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            padded_prompts = self.accelerator.pad_across_processes(
                repeated_prompts, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            
            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask"})
            
            if self.accelerator.is_main_process:
                # Decode the prompts and samples
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )
                
                rollout_score_time = time()
                
                # Compute rewards from the reward function
                all_scores, all_str_victim_outputs = self.reward_fn(
                    samples=all_str_samples, 
                    prompts=all_str_prompts, 
                    outputs=all_str_outputs,
                    return_texts=True,
                    **metadata
                )
                
                # Log generated texts
                self.train_text_logger.log(
                    all_str_prompts, 
                    all_str_outputs, 
                    all_str_victim_outputs,
                    all_scores
                )

                # Compute additional rewards
                # Self-BLEU rewards for diversity
                if self.config.method.bleu_reward_coef == 0:
                    all_bleu_scores = [0.] * len(all_scores)
                else:
                    if self.config.method.bleu_reward_include_prompts:
                        all_bleu_scores = self.bleu_reward_module(all_str_samples)
                        self.bleu_reward_module.append_reference(all_str_samples)
                    else:
                        all_bleu_scores = self.bleu_reward_module(all_str_outputs)
                        self.bleu_reward_module.append_reference(all_str_outputs)

                # Cosine similarity embedding rewards
                if self.config.method.cossimemb_reward_coef == 0:
                    all_cossimemb_scores = [0.] * len(all_scores)
                else:
                    if self.config.method.cossimemb_reward_include_prompts:
                        all_cossimemb_scores = self.cossimemb_reward_module(all_str_samples)
                        self.cossimemb_reward_module.append_reference(all_str_samples)
                    else:
                        all_cossimemb_scores = self.cossimemb_reward_module(all_str_outputs)
                        self.cossimemb_reward_module.append_reference(all_str_outputs)
                    
                # Textual similarity rewards
                if self.config.method.textual_sim_reward_coef == 0:
                    all_textualsim_scores = [0.] * len(all_scores)
                else:
                    if self.config.method.textual_sim_reward_include_prompts:
                        all_textualsim_scores = self.cossimemb_reward_module.compute_similarity(
                            all_str_prompts,
                            all_str_samples
                        )
                    else:
                        all_textualsim_scores = self.cossimemb_reward_module.compute_similarity(
                            all_str_prompts,
                            all_str_outputs
                        )
                        
                # Target embedding diversity rewards
                if self.config.method.target_sim_div_reward_coef == 0:
                    all_target_sim_div_scores = [0.] * len(all_scores)
                else:                    
                    all_target_sim_div_scores = self.cossimemb_reward_module.compute_l1_div_rewards(
                        all_str_victim_outputs
                    )
                
                # Gibberish penalty
                if self.config.method.giberish_penalty_coef == 0:
                    all_giberish_scores = [0.] * len(all_scores)
                else:
                    all_giberish_scores = self.giberish_penalty_penalty_module(all_str_outputs)
                
                # Aggregate all rewards
                all_scores = self._aggregate_traj_reward(
                    all_scores, all_bleu_scores, all_cossimemb_scores, 
                    all_textualsim_scores, all_target_sim_div_scores, all_giberish_scores, 
                    device
                )
                
                # Pad rewards and distribute to processes
                all_scores = pad_sequence(all_scores, batch_first=True, padding_value=-np.inf)
                max_len = torch.tensor(all_scores.shape[1], dtype=torch.long, device=device)

                stats["time/rollout_score"] = time() - rollout_score_time

                # Split scores for each process
                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1, max_len).unbind())
                self.history_scores += all_scores
            else:
                all_scores = None
                max_len = torch.tensor(0, dtype=torch.long, device=device)

            # Broadcast max_len to all processes
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(max_len, 0)
                
                # Receive scores from main process
                scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()
            
            # Create mask for valid score positions
            scores_mask = scores != -np.inf
            
            # For GRPO, we need to compute logprobs for each sample
            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors.repeat_interleave(self.num_generations, 0), 
                                                              samples, append_eos_token=True)
            
            # Tokenize outputs for computing logprobs
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == "seq2seq":
                # Add <pad> to the start of the output for seq2seq models
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            # Pad outputs to the same length
            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)
            
            # Compute logprobs for each sample
            # This is needed for GRPO's policy gradient computation
            repeated_prompts = prompt_tensors.repeat_interleave(self.num_generations, 0).to(device)
            
            # Store batch data for training
            grpo_batch = {
                'prompts': repeated_prompts,
                'responses': sample_outputs,
                'rewards': scores,
                'rewards_mask': scores_mask,
            }
            
            grpo_batch_data.append(grpo_batch)
            
            # Track statistics
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = (scores * scores_mask).sum(dim=1).mean(), (scores * scores_mask).sum(dim=1).std()
            
            all_scores_mean, all_scores_std = self.running_moments.update(torch.sum(scores * scores_mask, dim=1))
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()
            
            # Update the number of samples collected
            samples_collected += len(samples)
            
            stats["time/rollout_time"] = clock.tick()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {samples_collected} / {num_rollouts}]")
            tbar.update(min(len(samples), num_rollouts - (samples_collected - len(samples))))
        
        tbar.close()

        # Average statistics
        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in accumulated_stats[0]}
        self.accelerator.log(stats, step=iter_count)

        # Store batch data for training
        self.grpo_batch_data = grpo_batch_data

    def compute_loss(self, batch):
        """
        Compute GRPO loss based on the rewards and logprobs.
        
        GRPO uses direct policy gradients rather than PPO's clipped surrogate objectives.
        The core idea is to compute gradients directly with respect to rewards.
        
        Args:
            batch: Dictionary containing prompts, responses, rewards, and reward masks
            
        Returns:
            Dictionary of loss components and statistics
        """
        # Unpack batch
        prompts = batch['prompts']
        responses = batch['responses']
        rewards = batch['rewards']
        rewards_mask = batch['rewards_mask']
        
        device = prompts.device
        batch_size = prompts.shape[0]
        
        # Concatenate prompts and responses for causal models, or keep separate for seq2seq
        if self.config.model.model_arch_type == "seq2seq":
            # For seq2seq models, we compute logprobs on responses only
            attention_mask = prompts.not_equal(self.tokenizer.pad_token_id)
            decoder_attention_mask = responses.not_equal(self.tokenizer.pad_token_id)
            decoder_attention_mask[:, 0] = 1  # Always attend to first token
            
            # Compute logits for current policy
            outputs = self.model(
                input_ids=prompts,
                attention_mask=attention_mask if batch_size != 1 else None,
                decoder_input_ids=responses,
                decoder_attention_mask=decoder_attention_mask,
            )
            logits = outputs.logits
            
            # Compute logprobs of responses
            logprobs = logprobs_of_labels(logits[:, :-1, :], responses[:, 1:])
            
            # Compute KL if needed
            if self.config.method.loss_type == "kl":
                # Compute logits for reference model
                with torch.no_grad():
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            input_ids=prompts,
                            attention_mask=attention_mask if batch_size != 1 else None,
                            decoder_input_ids=responses,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            input_ids=prompts,
                            attention_mask=attention_mask if batch_size != 1 else None,
                            decoder_input_ids=responses,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                
                # Compute reference logprobs
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], responses[:, 1:])
                
                # Compute KL divergence
                kl_div = (logprobs - ref_logprobs) * decoder_attention_mask[:, :-1]
            
            # Apply reward mask to focus on specific tokens
            token_mask = decoder_attention_mask[:, :-1]
        else:
            # For causal models, we concatenate prompts and responses
            all_tokens = torch.cat((prompts, responses), dim=1)
            attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long()
            
            # Compute position ids for causal attention
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            # Forward pass through model
            logits, *_, _ = self.model(
                all_tokens, 
                attention_mask=attention_mask if batch_size != 1 else None,
                position_ids=position_ids
            )
            
            # Compute logprobs for next token prediction
            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            
            # Compute KL if needed
            if self.config.method.loss_type == "kl":
                # Compute logits for reference model
                with torch.no_grad():
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask if batch_size != 1 else None,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask if batch_size != 1 else None,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                
                # Compute reference logprobs
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])
                
                # Compute KL divergence
                kl_div = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            
            # Start from the end of prompt
            prompt_length = prompts.shape[1]
            token_mask = attention_mask[:, prompt_length-1:-1]
            logprobs = logprobs[:, prompt_length-1:]
            
            if self.config.method.loss_type == "kl":
                kl_div = kl_div[:, prompt_length-1:]
        
        # Compute rewards per token if needed
        if rewards.shape[1] == 1:
            # If we have a single reward per sequence, assign it to the last token
            per_token_rewards = torch.zeros_like(logprobs)
            for i in range(batch_size):
                # Find the last non-padding token
                last_token = token_mask[i].sum() - 1
                if last_token >= 0:
                    per_token_rewards[i, last_token] = rewards[i, 0]
        else:
            # If we have per-token rewards, use them directly
            per_token_rewards = rewards[:, :logprobs.shape[1]]
            # Ensure reward mask matches token mask
            token_mask = token_mask & rewards_mask[:, :logprobs.shape[1]]
        
        # Compute policy gradient loss
        if self.config.method.loss_type == "pg":
            # Standard policy gradient loss: -logprob * reward
            pg_loss = -logprobs * per_token_rewards
            pg_loss = (pg_loss * token_mask).sum() / token_mask.sum().clamp(min=1)
            loss = pg_loss
            stats = {
                "policy/loss": pg_loss.item(),
                "policy/entropy": (-logprobs * token_mask).sum().item() / token_mask.sum().clamp(min=1),
            }
        else:  # kl loss
            # KL-regularized policy gradient
            kl_loss = (kl_div * token_mask).sum() / token_mask.sum().clamp(min=1)
            reward_term = (-logprobs * per_token_rewards * token_mask).sum() / token_mask.sum().clamp(min=1)
            loss = kl_loss + reward_term
            stats = {
                "policy/loss": loss.item(),
                "policy/kl_loss": kl_loss.item(),
                "policy/reward_term": reward_term.item(),
                "policy/entropy": (-logprobs * token_mask).sum().item() / token_mask.sum().clamp(min=1),
            }
        
        # Add reward statistics
        masked_rewards = (per_token_rewards * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
        stats["rewards/mean"] = masked_rewards.mean().item()
        stats["rewards/std"] = masked_rewards.std().item()
        stats["rewards/max"] = masked_rewards.max().item()
        
        return loss, stats

    def step(self, iter_count: int):
        """
        Perform a single training step with GRPO.
        
        Args:
            iter_count: Current iteration count
            
        Returns:
            Dictionary of training statistics
        """
        if not hasattr(self, 'grpo_batch_data') or not self.grpo_batch_data:
            # If we don't have batch data, generate new experiences
            self.make_experience(self.config.train.batch_size, iter_count)
        
        stats = {}
        clock = Clock()
        
        # Process each batch
        for batch_idx, batch in enumerate(self.grpo_batch_data):
            # Compute loss and backward pass
            self.optimizer.zero_grad()
            
            loss, batch_stats = self.compute_loss(batch)
            
            # Update stats with batch stats
            for k, v in batch_stats.items():
                if k not in stats:
                    stats[k] = v
                else:
                    stats[k] += v
            
            # Backward pass and optimization
            self.accelerator.backward(loss)
            
            if self.config.optimizer.clip_grad:
                norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.optimizer.clip_grad
                )
                stats["policy/grad_norm"] = norm.item()
            
            self.optimizer.step()
            
            # Step the scheduler if needed
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Average stats across batches
        for k in list(stats.keys()):
            if k not in ["policy/grad_norm"]:
                stats[k] /= len(self.grpo_batch_data)
        
        # Clear batch data after processing
        self.grpo_batch_data = []
        
        # Add timing stats
        stats["time/step"] = clock.tick()
        
        return stats

    def evaluate(self):
        """Evaluate the model on the evaluation prompts"""
        logger.info("Evaluating model")
        
        # Handle sweep of generation parameters if configured
        if self.generate_sweep_kwarg is not None:
            gen_sweep_arg, gen_sweep_values = self.generate_sweep_kwarg
        else:
            gen_sweep_values = [None]

        desc = [
            f"generation sweep 0/{len(gen_sweep_values)}",
            f"eval batch 0/{len(self.eval_dataloader)}",
        ]
        tbar = logging.tqdm(
            total=len(self.eval_dataloader) * len(gen_sweep_values),
            desc=f"[{' | '.join(desc)}]",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        stats = {}
        table = []

        for i_sweep, gen_sweep_value in enumerate(gen_sweep_values):
            # A dedicated suffix for wandb logging
            if gen_sweep_value is not None:
                sweep_suffix = f"@{gen_sweep_arg}={gen_sweep_value}"
            else:
                sweep_suffix = ""

            all_samples = []
            all_prompts = []
            all_prompt_sizes = []
            all_metadata = []
            
            generate_time = time()
            for i_prompt, prompts in enumerate(self.eval_dataloader):
                metadata = {k: v for k, v in prompts.items() if k != "input_ids" and k != "attention_mask"}
                
                # Generate samples
                if self.generate_sweep_kwarg:
                    samples = self.generate_eval(
                        prompts["input_ids"], prompts["attention_mask"], **{gen_sweep_arg: gen_sweep_value}
                    )
                else:
                    samples = self.generate_eval(prompts["input_ids"], prompts["attention_mask"])

                # For seq2seq models, remove the first token
                if self.config.model.model_arch_type == "seq2seq":
                    samples = samples[:, 1:].contiguous()

                # Gather samples and prompts
                prompt_sizes = torch.tensor(prompts.input_ids.shape[1]).repeat(len(prompts.input_ids))
                prompts, samples, prompt_sizes = self.accelerator.gather_for_metrics(
                    self.accelerator.pad_across_processes(
                        [prompts.input_ids, samples, prompt_sizes.to(samples.device)],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                    )
                )
                
                all_samples.extend(samples.tolist())
                all_prompts.extend(prompts.tolist())
                all_prompt_sizes.extend(prompt_sizes.tolist())

                metadata = gather_dict(metadata, self.accelerator.gradient_state)
                all_metadata.append(metadata)

                # Update progress bar
                desc = [
                    f"generation sweep {i_sweep + 1}/{len(gen_sweep_values)}",
                    f"eval batch {i_prompt + 1}/{len(self.eval_dataloader)}",
                ]
                tbar.set_description(f"[{' | '.join(desc)}]")
                tbar.update()
            tbar.close()

            stats["time/generate"] = time() - generate_time

            if self.accelerator.is_main_process:
                # Decode the generated samples
                str_all_samples, str_all_prompts, str_all_outputs = self.decode(all_prompts, all_samples, all_prompt_sizes)
                
                # Process in batches to handle large evaluation sets
                eval_batch_size = self.config.train.batch_size
                for eval_batch_i in range(int(np.floor(len(str_all_samples) / eval_batch_size)) + 1):
                    # Get current batch
                    str_samples = str_all_samples[eval_batch_i*eval_batch_size:(eval_batch_i+1)*eval_batch_size]
                    str_prompts = str_all_prompts[eval_batch_i*eval_batch_size:(eval_batch_i+1)*eval_batch_size]
                    str_outputs = str_all_outputs[eval_batch_i*eval_batch_size:(eval_batch_i+1)*eval_batch_size]

                    # Apply human attacker templates if configured
                    if hasattr(self.config.model, "human_attacker_template_pool") and self.config.model.human_attacker_template_pool is not None:
                        attacker_template_pool = self.config.model.human_attacker_template_pool.split("\n")
                        attacker_template_pool = [v for v in attacker_template_pool if len(v) > 0]
                        attacker_template_samples = np.random.choice(attacker_template_pool, len(str_outputs)).tolist()
                        str_outputs_new = []
                        for v1, v2 in zip(str_prompts, attacker_template_samples):
                            if "<CONTEXT>" in v2:
                                str_out = v2.replace("<CONTEXT>", v1)
                            else:
                                str_out = v2
                            str_outputs_new.append(str_out)
                        str_outputs = str_outputs_new

                    # Set up columns for the results table
                    if eval_batch_i == 0:
                        columns = ["attacker prompt", "attacker output (victim prompt)"]
                    columns_data = [str_prompts, str_outputs]

                    # Collect metadata
                    metadata, *xs = all_metadata
                    for k in metadata:
                        for x in xs:
                            metadata[k].extend(x[k])

                    # Compute rewards if reward function is available
                    if self.reward_fn:
                        logger.info("Computing rewards")                   
                        rewards, victim_str_outputs = self.reward_fn(
                                samples=str_samples, 
                                prompts=str_prompts, 
                                outputs=str_outputs,
                                return_texts=True,
                                **metadata)

                        # Log evaluation results
                        self.eval_text_logger.log(
                            str_prompts,
                            str_outputs,
                            victim_str_outputs,
                            rewards,
                        )
                        
                        rewards = torch.tensor(
                            rewards,
                            dtype=float,
                        )

                        # Add victim outputs to columns
                        if eval_batch_i == 0:
                            columns.append("victim output")
                        columns_data.append([victim_str_output \
                                            for str_prompt, str_output, victim_str_output in 
                                                zip(str_prompts, str_outputs, victim_str_outputs)])
                    
                        # Add rewards to stats and columns
                        mean_reward = rewards.mean().item()
                        if eval_batch_i == 0:
                            columns.append("reward")
                        if not isinstance(rewards, list):
                            rewards = rewards.tolist()
                        columns_data.append(rewards)
                        stats[f"reward/mean{sweep_suffix}"] = mean_reward
                    
                    # Compute additional metrics if metric function is available
                    if self.metric_fn:
                        logger.info("Computing metrics")
                        metric_time = time()
                        metrics = self.metric_fn(samples=str_samples, prompts=str_prompts, outputs=str_outputs, **metadata)
                        stats["time/metric"] = time() - metric_time

                        # Add metrics to stats
                        mean_metrics = {
                            f"metrics/{k}{sweep_suffix}": torch.as_tensor(xs).mean(-1).item() for k, xs in metrics.items()
                        }
                        stats.update(mean_metrics)

                        # Add metrics to columns
                        for metric, values in metrics.items():
                            # Skip metrics that are scalers since they represent aggregated values
                            if isinstance(values, float):
                                continue
                            columns.append(metric)
                            if not isinstance(values, list):
                                values = values.tolist()
                            columns_data.append(values)

                    # Add generation sweep parameter if configured
                    if self.generate_sweep_kwarg:
                        columns.insert(0, gen_sweep_arg)
                        columns_data.insert(0, [gen_sweep_value] * len(str_samples))

                    # Add this batch to the results table
                    table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        logger.info("Summarizing evaluation")
        if self.accelerator.is_main_process:
            # Flatten table rows
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            # Create and display the results table
            rich_table = Table(*columns, title=table_title, show_lines=True)
            for ix in range(max(min(3, len(rows)), len(gen_sweep_values))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])
            Console().print(rich_table)

            # Log to wandb if configured
            if self.config.train.tracker == "wandb":
                import wandb
                stats["samples"] = wandb.Table(columns, rows)

        self.nth_evaluation += 1
        return stats