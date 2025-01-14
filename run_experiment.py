import os
os.environ['NUMEXPR_MAX_THREADS'] = '1'

import logging
import numexpr as ne
import numpy as np
import torch
import datetime
from ddopai.envs.pricing.dynamic import DynamicPricingEnv
from ddopai.envs.actionprocessors import ClipAction, RoundAction
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

def main():

    logging_level = logging.INFO
    logging.basicConfig(level=logging_level)

    ne.set_num_threads(1)
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)

    set_warnings(logging.INFO) # turn off warnings for any level higher or equal to the input level

    PROJECT_NAME = "dynamic_pricing"
    LIBRARIES_TO_TRACK = ["ddopai", "mushroom_rl"]
    ENVCLASS = DynamicPricingEnv
    RESULTS_DIR = "results"

    logging.info("######################### Preparing Experiment #################################")

    config_train, config_agent, config_env, AgentClass, agent_name = prep_experiment(
        PROJECT_NAME,
        LIBRARIES_TO_TRACK,
        config_train_name="config_train.yaml",
        config_agent_name="config_agent.yaml",
        config_env_name="config_env.yaml",
    )

    logging.info("######################### Getting data #########################################")
    raw_data, val_index_start, test_index_start = get_online_data(
        config_env,
        overwrite=False
    )

    logging.info("######################### Setting up environment ###############################")
    round_action = RoundAction(unit_size=config_env["unit_size"])
    postprocessors = [round_action]

    environment = set_up_env(ENVCLASS, raw_data, val_index_start, test_index_start, config_env, postprocessors)

    logging.info("######################### Setting up agent #####################################")

    logging.info(f"Agent: {agent_name}")


    if AgentClass.train_mode == "env_interaction":
        if "link" in config_agent:
            glm_link, price_function = set_up_agent(config_agent["link"])
            config_agent["g"] = glm_link
            config_agent["price_function"] = price_function
            del config_agent["link"]
        agent = AgentClass(
            environment_info=environment.mdp_info,
            **config_agent
        )

    else:
        raise ValueError("Invalid train_mode for online training")

    logging.info("######################### Preparing training ###################################")

    earlystoppinghandler = set_up_earlystoppinghandler(config_train)

    logging.info("######################### Training agent #######################################")

    run_experiment(
        agent,
        environment,
        n_epochs=config_train["n_epochs"],
        n_steps=config_train["n_steps"],
        early_stopping_handler=earlystoppinghandler,
        save_best=config_train["save_best"],
        run_id=wandb.run.id,
        tracking="wandb",
        eval_step_info=False,
        print_freq=1,
        results_dir = RESULTS_DIR
    )

    

    logging.info("######################### Finished Script ######################################")

if __name__ == "__main__":
    main()