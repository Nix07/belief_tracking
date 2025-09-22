import logging
import os
from typing import Union

import yaml

# Configure logger
logger = logging.getLogger("mind")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

PROJECT_ROOT = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])

try:
    with open(os.path.join(PROJECT_ROOT, "env.yml"), "r") as f:
        config = yaml.safe_load(f)
        DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, config["DATA_DIR"])

except FileNotFoundError:
    logger.error(
        f'''env.yml not found in {PROJECT_ROOT}!
Setting MODEL_ROOT="". Models will now be downloaded to conda env cache, if not already there
Other defaults are set to:
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    HPARAMS_DIR = "hparams"'''
    )
    DEFAULT_MODELS_DIR = ""
    DEFAULT_DATA_DIR = "data"


def load_env_var(var: str) -> Union[str, None]:
    try:
        PROJECT_ROOT = "/".join(
            os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]
        )
        with open(os.path.join(PROJECT_ROOT, "env.yml"), "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"""env.yml not found in {PROJECT_ROOT}!""")

    if var not in config or config[var] is None or config[var] == "":
        logger.error(f"{var} not set in env.yml!")
        return None

    else:
        return config[var]
