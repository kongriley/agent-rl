# Red-teaming with GRPO (Gradient-based Reinforcement Learning with Policy Optimization)

This repository contains an implementation of GRPO (Gradient-based Reinforcement Learning with Policy Optimization) for red-teaming language models. GRPO is an alternative to PPO that uses direct policy gradient optimization.

## Key Differences Between GRPO and PPO

1. **Direct Policy Optimization**: GRPO uses direct policy gradients rather than PPO's clipped surrogate objectives.
2. **Multiple Generations**: GRPO generates multiple samples per prompt to better estimate policy gradients.
3. **Loss Formulation**: GRPO provides flexibility in loss functions, supporting both KL-regularized losses and standard policy gradient losses.
4. **Simplified Training Loop**: GRPO uses a more direct optimization approach without the need for complex advantage estimation.

## Configuration

The GRPO trainer requires a YAML configuration file. A sample configuration is provided in `sample_redteam_grpo_config.yaml`. The key GRPO-specific parameters are:

```yaml
method:
  name: "RedteamGRPOConfig"
  # GRPO specific parameters
  num_generations: 4        # Number of generations per prompt
  temperature: 1.0          # Temperature for sampling
  top_p: 1.0                # Top-p sampling parameter
  loss_type: "kl"           # Loss type: "kl" or "pg"
  
  # Reward components
  bleu_reward_coef: -0.5    # Self-BLEU penalty coefficient (negative for penalty)
  cossimemb_reward_coef: 0.0  # Cosine similarity reward coefficient
  textual_sim_reward_coef: 0.0  # Textual similarity reward coefficient
  # ... other reward components
```

## Usage

To run the GRPO trainer, use the `run_redteam_grpo.py` script:

```bash
python run_redteam_grpo.py \
  --config_path sample_redteam_grpo_config.yaml \
  --reward_fn path/to/your/reward_function.py \
  --output_dir outputs/grpo_redteam \
  --seed 42
```

## Implementing a Reward Function

The reward function should have the following signature:

```python
def reward_fn(samples, prompts, outputs, return_texts=False, **kwargs):
    """
    Compute rewards for the generated outputs.
    
    Args:
        samples: List of full samples (prompt + output)
        prompts: List of prompts
        outputs: List of generated outputs
        return_texts: Whether to return the victim model outputs as texts
        **kwargs: Additional arguments
        
    Returns:
        rewards: List of rewards for each sample
        victim_outputs: List of victim model outputs (only if return_texts=True)
    """
    # Your implementation here
    # ...
    
    if return_texts:
        return rewards, victim_outputs
    else:
        return rewards
```

You should create a `get_reward_fn()` function in your reward module that returns this reward function.

## Logs and Outputs

The trainer creates the following logs:

1. Training samples and their rewards in `train.csv`
2. Evaluation samples and their rewards in `eval.csv`
3. Model checkpoints in the output directory
4. WandB logs if configured

## Example Reward Components

The GRPO trainer supports various reward components that can be combined:

1. **Primary Reward**: The main reward signal from the reward function (e.g., victim model behavior)
2. **Self-BLEU Penalty**: Encourages diversity by penalizing repetitive outputs
3. **Cosine Similarity Rewards**: Uses sentence embeddings to measure and reward semantic diversity
4. **Textual Similarity**: Measures similarity between prompts and responses
5. **Target Embedding Diversity**: Encourages diverse responses from the victim model
6. **Gibberish Penalty**: Penalizes outputs that are detected as gibberish

## Implementation Details

The GRPO implementation is in `accelerate_redteam_grpo_trainer.py`. The key components are:

1. `make_experience`: Generates experiences using the model
2. `compute_loss`: Computes the GRPO loss based on rewards and logprobs
3. `step`: Performs a single training step
4. `evaluate`: Evaluates the model on the evaluation prompts

The implementation supports both causal language models and seq2seq models.

## Requirements

- PyTorch
- Transformers
- Accelerate
- TRL/TRLX library
- NLTK (for BLEU tokenization)
- WandB (optional, for logging) 