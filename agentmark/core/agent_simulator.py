"""
Model interaction module (Agent Simulator).
Responsibilities: wrap all LLM API interaction logic.
"""

from .prompt_utils import format_behaviors_list, generate_behaviors_example
from .parser_utils import extract_probabilities


def get_behavior_probabilities(client, model, role_config, event, behaviors, probability_template):
    """
    Get a probability distribution over behaviors given an event.

    Args:
        client: OpenAI client instance
        model: Model name
        role_config: Role config (name, profile, system_prompt)
        event: Formatted event text
        behaviors: Behavior types list, e.g. ['like', 'favorite', 'share', ...]
        probability_template: Prompt template for probability estimation

    Returns:
        tuple: (probabilities_dict, raw_response_text)
            - probabilities_dict: Probability dict, e.g. {'like': 0.3, 'favorite': 0.2, ...}
            - raw_response_text: Raw API response text
    """
    name = role_config['name']
    profile = role_config['profile']
    
    # Build probability prompt
    probability_prompt = probability_template.format(
        name=name,
        event=event,
        behaviors=format_behaviors_list(behaviors),
        behaviors_example=generate_behaviors_example(behaviors)
    )
    
    print("Probability prompt:")
    print(probability_prompt)
    
    # Call API to get behavior probabilities
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": role_config["system_prompt"].format(name=name, profile=profile)
            },
            {
                "role": "user", 
                "content": probability_prompt
            }
        ]
    )
    
    # Get response text
    response_text = response.choices[0].message.content
    print("\nProbability response:")
    print(response_text)
    
    # Extract probabilities
    probabilities = extract_probabilities(response_text, behaviors)
    
    if probabilities:
        print("\nExtracted probabilities:")
        print(probabilities)
    else:
        print("\nWarning: failed to extract probabilities")
    
    return probabilities, response_text


def get_behavior_description(client, model, role_config, event, behavior, behavior_template):
    """
    Get a detailed description for a selected behavior given an event.

    Args:
        client: OpenAI client instance
        model: Model name
        role_config: Role config
        event: Formatted event text
        behavior: Selected behavior, e.g. "like"
        behavior_template: Prompt template for behavior description

    Returns:
        str: Model-generated behavior description
    """
    name = role_config['name']
    profile = role_config['profile']
    
    # Build behavior description prompt
    behavior_prompt = behavior_template.format(
        name=name,
        event=event,
        behavior=behavior
    )
    
    print(f"\nBehavior prompt (behavior: {behavior}):")
    print(behavior_prompt)
    
    # Call API for behavior description
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": role_config["system_prompt"].format(name=name, profile=profile)
            },
            {
                "role": "user", 
                "content": behavior_prompt
            }
        ]
    )
    
    behavior_description = response.choices[0].message.content
    print("\nBehavior description:")
    print(behavior_description)
    
    return behavior_description
