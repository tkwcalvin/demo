"""
Utility Functions Module
=======================

This module contains utility functions for text processing, code extraction,
and various helper operations used throughout the code generation system.

Key functions:
- Text manipulation: split_and_replace_with_random_words, split_and_remove_chunk
- Code extraction: response_2_code, response_2_code_if_no_text
- String processing: get_ith_element, string_to_int
- Random word generation: generate_random_common_word

Features:
- Text corruption for robustness testing
- Code block extraction from model responses
- String parsing and manipulation utilities
"""

import random
import re
from config import common_words


# ============================================================================
# RANDOM WORD GENERATION
# ============================================================================

def generate_random_common_word():
    """Generate a random common word from the predefined list."""
    return random.choice(common_words)

# ============================================================================
# TEXT CORRUPTION FUNCTIONS
# ============================================================================
# These functions are used to create modified versions of problem descriptions
# for testing model robustness to incomplete or corrupted input

def split_and_replace_with_random_words(text, percentage):
    """
    Replace a percentage of words in the text with random common words.
    
    This function is used to create corrupted versions of problem descriptions
    to test how models handle incomplete or misleading information.
    
    Args:
        text (str): The input text to modify
        percentage (int): Percentage of words to replace (0-100)
    
    Returns:
        str: Modified text with random words replacing the specified percentage
    """
    # Split the input string into words
    words = text.split()
    num_words = len(words)

    # Calculate the number of words to replace based on percentage
    num_words_to_replace = calculate_percentage_integer(num_words, percentage)

    if num_words_to_replace == 0:
        return text  # If there are too few words to replace, return the original text

    # Choose a random starting index for the chunk to replace
    start_index = random.randint(0, num_words - num_words_to_replace)
    end_index = start_index + num_words_to_replace

    # Generate random common words to replace the chunk
    random_common_words = [generate_random_common_word() for _ in range(num_words_to_replace)]

    # Create the modified word list
    modified_words = words[:start_index] + random_common_words + words[end_index:]
    modified_text = ' '.join(modified_words)

    return modified_text

def split_and_remove_chunk(text, percentage):
    """
    Remove a percentage of words from the text.
    
    This function creates incomplete versions of problem descriptions
    to test how models handle missing information.
    
    Args:
        text (str): The input text to modify
        percentage (int): Percentage of words to remove (0-100)
    
    Returns:
        str: Modified text with the specified percentage of words removed
    """
    # Split the input string into words
    words = text.split()
    num_words = len(words)

    # Calculate the number of words to remove based on percentage
    num_words_to_remove = calculate_percentage_integer(num_words, percentage)

    if num_words_to_remove == 0:
        return text  # If there are too few words to remove, return the original text

    # Choose a random starting index for the chunk to remove
    start_index = random.randint(0, num_words - num_words_to_remove)
    end_index = start_index + num_words_to_remove

    # Create a list to store the words to keep
    words_to_keep = []

    # Iterate through the words and keep those outside the removal chunk
    for index, word in enumerate(words):
        if index < start_index or index >= end_index:
            words_to_keep.append(word)

    # Join the remaining words to create the modified text
    modified_text = ' '.join(words_to_keep)

    return modified_text

def calculate_percentage_integer(value, percentage):
    """
    Calculate a percentage of a value and return as integer.
    
    Args:
        value (int): The base value
        percentage (int): The percentage (0-100)
    
    Returns:
        int: The calculated percentage, rounded to nearest integer
    """
    result = value * (percentage / 100.0)
    rounded_result = round(result)
    return rounded_result






# ============================================================================
# CODE EXTRACTION FUNCTIONS
# ============================================================================
# These functions extract Python code from model responses that contain markdown code blocks

def response_2_code(response):
    """
    Extract the first code block from a model response.
    
    This function finds the first triple backtick code block in the response
    and returns the code content without the markdown formatting.
    
    Args:
        response (str): The model response containing code blocks
    
    Returns:
        str: The extracted code content, or empty string if no code block found
    """
    code_template = re.compile('```.*\n([\s\S]+?)\n```', re.M)
    code = code_template.findall(response)
    if len(code) > 0:
        return code[0]  # Return the first code block
    else:
        return ''

def response_2_code_if_no_text(response):
    """
    Extract code only if the response consists solely of a code block.
    
    This function is more strict than response_2_code - it only extracts code
    if the entire response is a single code block with optional whitespace.
    This helps distinguish between responses that are pure code vs. responses
    that contain code along with explanatory text.
    
    Args:
        response (str): The model response to analyze
    
    Returns:
        str: The extracted code content if response is pure code block,
             empty string otherwise
    """
    # Regular expression that matches only responses that are entirely code blocks
    code_template = re.compile(r'^\s*```.*?\n([\s\S]+?)\n```\s*$', re.M)
    match = code_template.match(response)
    if match:
        return match.group(1)  # Return the code block content
    return ''  # Return empty string if response contains other text







# ============================================================================
# STRING PROCESSING UTILITIES
# ============================================================================

def get_ith_element(input_string, i):
    """
    Extract the i-th element from a string split by underscores.
    
    This function is commonly used to parse option strings like "randRemove25"
    where the underscore-separated parts have specific meanings.
    
    Args:
        input_string (str): The input string to split
        i (int): The index of the element to extract (0-based)
    
    Returns:
        str: The i-th element, or empty string if index is out of range
    """
    # Split the input string by '_' to create a list of elements
    elements = input_string.split('_')

    # Check if i is a valid index within the range of elements
    if 0 <= i < len(elements):
        return elements[i]
    else:
        return ""  # Return empty string if index is out of range

def string_to_int(input_string):
    """
    Safely convert a string to an integer.
    
    Args:
        input_string (str): The string to convert
    
    Returns:
        int or None: The converted integer, or None if conversion fails
    """
    try:
        result = int(input_string)
        return result
    except ValueError:
        return None  # Return None if the string cannot be converted to an integer