# FILE: run_experiment.py
import os
os.environ['NUMEXPR_MAX_THREADS'] = '1'

import logging
import numexpr as ne
import numpy as np
import torch
import datetime
from ddopai.envs.pricing.dynamic import DynamicPricingEnv
from ddopai.envs.pricing.dynamic_RL2 import RL2DynamicPricingEnv
from ddopai.envs.pricing.dynamic_lag import LagDynamicPricingEnv
from ddopai.envs.pricing.dynamic_inventory import DynamicPricingInvEnv
from ddopai.envs.actionprocessors import ClipAction, RoundAction
from ddopai.agents.obsprocessors import ConvertDictSpace
from ddopai.experiments.experiment_functions_online import run_experiment
from ddopai.experiments.meta_experiment_functions import *
import requests
import yaml
import re
import pandas as pd
import wandb
from copy import deepcopy
import warnings
import gc
from mushroom_rl import core 
import pickle
import argparse
import sys
import traceback
import random

logging_level = logging.INFO
logging.basicConfig(level=logging_level)

LIBRARIES_TO_TRACK = ["ddopai", "mushroom_rl"]
RESULTS_DIR = "results"

def get_ENVCLASS(class_name):
    if class_name == "DynamicPricingEnv":
        return DynamicPricingEnv
    elif class_name == "DynamicPricingInvEnv":
        return DynamicPricingInvEnv
    elif class_name == "LagDynamicPricingEnv":
        return LagDynamicPricingEnv
    elif class_name == "RL2DynamicPricingEnv":
        return RL2DynamicPricingEnv
    else:
        raise ValueError(f"Unknown class name {class_name}")

def combine_dict(dict1: dict, dict2: dict) -> dict:
    """
    Combines two dictionaries. Raises error if there are overlapping keys.
    """
    overlapping_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    if overlapping_keys:
        raise ValueError(f"Overlapping keys detected: {overlapping_keys}")
    return {**dict1, **dict2}

def convert_str_none_to_none(d: dict) -> dict:
    """Recursively convert all string "None" in a dictionary to Python's None."""
    for key, value in d.items():
        if value == "None":
            d[key] = None
        elif isinstance(value, dict):
            convert_str_none_to_none(value)
    return d

def merge_with_namespace(target_dict: dict, source_dict: dict, target_dict_name: str) -> dict:
    """
    Merge source_dict items into target_dict based on key namespace filtering.
    
    For each key in source_dict, if it starts with target_dict_name plus a hyphen,
    the rest of the keys will be used to navigate target_dict. If the keys exist,
    then value is overwritten.
    """
    for namespaced_key, value in source_dict.items():
        keys = namespaced_key.split('-')
        if keys[0] != target_dict_name:
            continue
        keys = keys[1:]
        d = target_dict
        exists = True
        # Iterate over the keys until the last one
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                exists = False
                break
            d = d[key]
        # final key check
        if exists and keys[-1] in d:
            prev = d[keys[-1]]
            d[keys[-1]] = value
            logging.info(f"Overwriting key {namespaced_key}: {prev} -> {value}")
        else:
            raise ValueError(f"Key {namespaced_key} not found in {target_dict_name}.")
    return target_dict

def run_pipeline(sweep_config, project_name="pricing_cMDP", config_env="config_env.yaml", 
                 config_train="config_train.yaml", config_agent="config_agent.yaml"):
    """
    Run the pipeline:
    1. Setup wandb
    2. Create agent
    3. Get parameters for env and training
    4. Run experiment
    5. Save results
    """
    ne.set_num_threads(1)
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)
    

    
    config_train = import_config(config_train)
    config_agent = import_config(config_agent) # General config file containing all agent parameters
    config_env = import_config(config_env)
    
    
    wandb.config.update(config_train)
    wandb.config.update(config_agent)
    wandb.config.update(config_env)
    
    if sweep_config:
        merge_with_namespace(config_train, wandb.config._items, "config_train")
        merge_with_namespace(config_env, wandb.config._items, "config_env")
        merge_with_namespace(config_agent, wandb.config._items, "config_agent")
        
    config_env = convert_str_none_to_none(config_env)
    config_train = convert_str_none_to_none(config_train)
    config_agent = convert_str_none_to_none(config_agent)
    
    AgentClass = select_agent(config_train["agent"]) # Select agent class and import dynamically
    agent_name = config_train["agent"]
    config_agent = config_agent[config_train["agent"]]
    n_steps_per_fit = config_agent.pop("n_steps_per_fit", None)
    n_episodes_per_fit = config_agent.pop("n_episodes_per_fit", None)
    
    if config_env['lag_window_params'].get("lag_window") is not None:
        for env_kwargs in config_env["env_kwargs"]:
            env_kwargs["lag_window"] = config_env['lag_window_params']['lag_window']
            env_kwargs["env_class"] = "LagDynamicPricingEnv"
    if "gamma" in config_agent:
        for env_kwargs in config_env["env_kwargs"]:
            env_kwargs["gamma"] = config_agent["gamma"]
        del config_agent["gamma"]
        
    if agent_name == "RL2PPO":
        for env_kwargs in config_env["env_kwargs"]:
            env_kwargs["env_class"] = "RL2DynamicPricingEnv"

    artifact = wandb.use_artifact(sweep_config['artifact'])
    path = artifact.download()
    raw_data_path = os.path.join(path, 'raw_data.pkl')
    with open(raw_data_path, 'rb') as f:
        raw_data = pickle.load(f)
    # setting the random seeds 
    np.random.seed(artifact.metadata["seed"])
    torch.manual_seed(artifact.metadata["seed"])
    random.seed(artifact.metadata["seed"])
    wandb.config.update({"seed": artifact.metadata["seed"]})
    #
    round_action = RoundAction(unit_size=config_env["unit_size"])
    postprocessors = [round_action]
    environments = prepare_env_online(
        get_ENVCLASS=get_ENVCLASS,
        raw_data=raw_data,
        val_index_start=0,
        test_index_start=0,
        config_env=config_env,
        postprocessors=postprocessors
    )

    logging.info(f"Agent: {agent_name}")

    if AgentClass.train_mode == "env_interaction":
        if agent_name in ["SAC", "PPORNN", "RL2PPO"]:
            obsprocessors = [ConvertDictSpace(keep_time_dim=False, )]
        else:
            obsprocessors = []
        if "link" in config_agent:
            glm_link, price_function = set_up_agent(AgentClass, environments[0], config_agent)
            config_agent["g"] = glm_link
            config_agent["price_function"] = price_function
            del config_agent["link"]
        if agent_name == "Clairvoyant":
            agent = AgentClass(
                alpha=environments[0].alpha,
                beta=environments[0].beta,
                environment_info=environments[0].mdp_info,
                obsprocessors=obsprocessors,
                **config_agent
            )
        else:
            agent = AgentClass(
                environment_info=environments[0].mdp_info,
                obsprocessors=obsprocessors,
                **config_agent
            )
    else:
        raise ValueError("Invalid train_mode for online training")

    earlystoppinghandler = set_up_earlystoppinghandler(config_train)

    dataset = run_experiment(
        agent=agent,
        train=environments,
        n_epochs=config_train["n_epochs"],
        n_steps=config_train["n_steps"],
        n_steps_per_fit=n_steps_per_fit,
        n_episodes_per_fit=n_episodes_per_fit,
        early_stopping_handler=earlystoppinghandler,
        save_best=config_train["save_best"],
        run_id=wandb.run.id,
        tracking="wandb",
        eval_step_info=False,
        print_freq=1,
        results_dir=RESULTS_DIR
    )

    wandb.finish()

def train():
    project_name = "pricing_cMDP"
    wandb.init(
         project=project_name,
         name=f"{project_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_pipeline(sweep_config=wandb.config)

def main(sweep_id, max_runs, project_name):
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise ValueError("WandB API key not set in environment variables.")
    wandb.login(key=wandb_key)

    wandb.agent(sweep_id, function=train, count=max_runs, project=project_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--sweep_id', type=str, default=None, help='ID for the sweep.')
    parser.add_argument('--max_runs', type=int, default=None, help='Maximum number of runs.')
    parser.add_argument('--project_name', type=str, default=None, help='Project name')
    logging.info("Arguments parsed.")
    args = parser.parse_args()

    try:
        main(args.sweep_id, args.max_runs, args.project_name)
    except Exception as e:
        logging.exception("An error occurred during experiment execution.")
        sys.exit(1)