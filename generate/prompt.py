"""
Prompt Management Module
======================

This module handles prompt loading from configuration files and prompt creation
for different experiment types and phases.

Key functions:
- load_prompt_from_config: Load prompt templates from config.yaml
- create_prompt: Create prompts for different experiment options

Features:
- YAML configuration support for prompt templates
- Phase-based prompt selection
- Support for different experiment types (original, randRemove, manualRemove)
"""

from utils import split_and_remove_chunk, split_and_replace_with_random_words
from config import PROMPT_START_3, PROMPT_START_3_v2
from main import args
import os
import yaml

# Load configuration from YAML file
config = []
try:
    with open(os.path.join('config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
except Exception:
    print("cannot find config.yaml!!")

def load_prompt_from_config(phase):
    """
    Load prompt template from config.yaml based on experiment phase.
    
    This function loads prompt templates from the YAML configuration file
    based on the specified phase (1 or 2) and the prompt ID specified
    in command line arguments.
    
    Args:
        phase (int): The experiment phase (1 or 2)
                     - Phase 1: Initial code generation or question asking
                     - Phase 2: Question evaluation and answer generation
    
    Returns:
        str: The loaded prompt template
    
    Raises:
        SystemExit: If phase is invalid, prompt ID not found, or loading fails
    """
    # Determine which prompt ID to use based on phase
    if(phase == 1):
        prompt_id = args.phase1_prompt
    elif(phase == 2):
        prompt_id = args.phase2_prompt
    else:
        print(f'Invalid phase number passed to "load_prompt_from_config" function')
        raise SystemExit(1)
    
    try:
        # Load the prompt from the configuration
        if prompt_id in config[f'phase{phase}_prompts'].keys():
            loaded_prompt = config[f'phase{phase}_prompts'][prompt_id]
            return loaded_prompt
        else:
            print(f'"{prompt_id}" does not exist in the list of phase{phase} prompts in config.yaml file.')
            raise SystemExit(1)
    except Exception as e:
        print(f'Failed to load phase{phase} prompt, exception: ', e)
        raise SystemExit(1)


def create_prompt(description, option='original', percentage=0):
    """
    Create a prompt based on the experiment option and problem description.
    
    This function creates different types of prompts based on the experiment option:
    - original: Standard prompt with unmodified description
    - randRemove: Prompt with randomly removed portions of the description
    - manualRemove: Prompt with manually modified description (work in progress)
    - Other options: Prompt with random word replacements
    
    Args:
        description (str): The problem description to use in the prompt
        option (str): The experiment option type
        percentage (int): Percentage of text to modify (for randRemove options)
    
    Returns:
        str: The complete prompt combining the template with the (possibly modified) description
    """
    if option == 'original':
        # Standard prompt with unmodified description
        prompt = PROMPT_START_3
        return prompt + description
    elif option.startswith('randRemove'):
        # Remove a percentage of words from the description randomly
        return PROMPT_START_3 + split_and_remove_chunk(description, percentage)
    elif option.startswith('manualRemove'):
        # Use different prompt template for manually modified descriptions
        # TODO(jwu): This is work in progress
        return PROMPT_START_3_v2 + description
    else:
        # Default case: replace words with random common words
        return PROMPT_START_3 + split_and_replace_with_random_words(description, percentage)