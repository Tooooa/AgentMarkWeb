"""
Prompt utility module.
Purpose: formatting helpers for LLM prompt generation.
"""


def format_behaviors_list(behaviors):
    """
    Format a list of behaviors as a bullet list string.

    Args:
        behaviors (list): Behavior list.

    Returns:
        str: Formatted behavior list string.

    Example:
        >>> behaviors = ["like", "save", "share"]
        >>> format_behaviors_list(behaviors)
        '- like\\n- save\\n- share'
    """
    return "\n".join([f"- {behavior}" for behavior in behaviors])


def generate_behaviors_example(behaviors):
    """
    Generate an example JSON string for behavior probabilities.

    Args:
        behaviors (list): Behavior list.

    Returns:
        str: Example JSON string.

    Example:
        >>> behaviors = ["like", "save", "share"]
        >>> generate_behaviors_example(behaviors)
        '{ "like": <probability of like>, "save": <probability of save>, "share": <probability of share> }'
    """
    example_dict = {behavior: f"<probability of {behavior}>" for behavior in behaviors}
    return "{ " + ", ".join([f'"{k}": {v}' for k, v in example_dict.items()]) + " }"
