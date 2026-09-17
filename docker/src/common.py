from enum import Enum
import json
import re
import os

command_to_index = {
    "STOP" : 1,
    "START": 2,
    "ENGINE_STOP" : 3,
    "ENGINE_START" : 4,
    "CRANK_REQUEST" : 5,
    "STOP_IN_PITLANE_TOGGLE_ON" : 6,
    "STOP_IN_PITLANE_TOGGLE_OFF" : 7,
    "SPEED_BY_RC_TOGGLE_ON" : 8,
    "SPEED_BY_RC_TOGGLE_OFF" : 9,
    "FLAG_BY_RC_TOGGLE_ON" : 10,
    "FLAG_BY_RC_TOGGLE_OFF" : 11,
    "INTO_PITLANE_TOGGLE_ON" : 12,
    "INTO_PITLANE_TOGGLE_OFF" : 13,
    "JOYSTICK_TOGGLE_ON" : 14,
    "JOYSTICK_TOGGLE_OFF" : 15,
    "SPEED_REQUEST" : 16,
    "GG_SCALE" : 17
}

index_to_command = {v: k for k, v in command_to_index.items()}

def is_command_classification_correct(llm_answer: str, correct_answer: list[2], probability_threshold = None, version = 1) -> tuple:
    if version == 1 and probability_threshold is not None:
        return is_command_classification_correct_1(llm_answer, correct_answer, probability_threshold)
    elif version == 2 and probability_threshold is None:
        return is_command_classification_correct_2(llm_answer, correct_answer)
    else:
        raise ValueError("Unsupported version or missing probability threshold. Supported versions are 1 and 2, with a probability threshold for version 1.")
    
def is_command_classification_correct_1(llm_answer: str, correct_answer: list[2], probability_threshold: float) -> tuple:
    correct_request = correct_answer[0]
    correct_value = correct_answer[1]

    # Check whether the output is in the json format as required
    try:
        llm_answer_json = json.loads(llm_answer)
    except:
        return ("WRONG_OUTPUT_FORMAT",)

    # Process the json
    max_probability = 0
    llm_request_prediction = None
    llm_value_prediction = None
    try: 
        # Find the command of highest probability
        for raw_command in llm_answer_json.keys():
            curr_probability = float(llm_answer_json[raw_command]["probability"])
            if curr_probability > max_probability:
                llm_request_prediction = str(raw_command)
                llm_value_prediction = None
                try:
                    llm_value_prediction = int(llm_answer_json[raw_command]["value"]) if llm_answer_json[raw_command]["value"] is not None else None
                except:
                    return ("INCORRECT_VALUE", llm_request_prediction, llm_value_prediction, max_probability)
                max_probability = curr_probability
    except:
        return ("WRONG_OUTPUT_FORMAT",)
    
    # Check the correctness of llm's answer
    if llm_request_prediction == correct_request and llm_value_prediction == correct_value:
        # Check the probability threshold
        if max_probability < probability_threshold:
            return ("TOO_LOW_PROBABILITY", llm_request_prediction, llm_value_prediction, max_probability)
        return ("CLASSIFICATION_CORRECT", llm_request_prediction, llm_value_prediction, max_probability)
    elif llm_request_prediction == correct_request:
        return ("INCORRECT_VALUE", llm_request_prediction, llm_value_prediction, max_probability)
    elif llm_value_prediction == correct_value:
        return ("INCORRECT_REQUEST", llm_request_prediction, llm_value_prediction, max_probability)
    else:
        return ("CLASSIFICATION_INCORRECT", llm_request_prediction, llm_value_prediction, max_probability)

def is_command_classification_correct_2(llm_answer, correct_answer) -> str:
        if llm_answer == correct_answer:
            return "CLASSIFICATION_CORRECT"
        else:
            return "CLASSIFICATION_INCORRECT"

def get_test_dataset(source_file_path: str) -> tuple:
    # read basic commands from json
    with open(source_file_path) as file:
        basic_commands = json.load(file)

    # read prompt template
    with open('prompt_template_1.txt') as file:
        prompt_template = file.read()

    X = [] # prompts
    y = [] # labels (raw commands)

    for raw_command in basic_commands.keys():
        for user_command in basic_commands[raw_command]:
            value = user_command[1] if type(user_command) is list else None
            user_command = user_command[0] if type(user_command) is list else user_command
            X.append([user_command, prompt_template.replace("<user_command>", user_command)])
            y.append([raw_command, value])
    
    return (X, y)

def json_to_list_of_tuples(json, number_of_supported_commands) -> list:
    
    tuples = []

    for key in json.keys():
        if key in command_to_index.keys():
            tuples.append((command_to_index[key], json[key]["value"], json[key]["probability"]))

    if len(tuples) != number_of_supported_commands:
        raise Exception("Number of commands in the json is different from the number of supported commands")

    return tuples
    
def get_prompt_template(command_list=list(), prompt_version = 1) -> str:
    if prompt_version == 1:
       with open(os.path.join(os.path.dirname(__file__), 'prompt_template_1.txt'), 'r') as file:
            lines = file.readlines()
            header = lines[0:2]
            command_definitions = lines[2:19]
            footer = ("".join(lines[19:30])
            + ",\n".join([
            f'''    "{command}": {{
            "value": {"null" if command not in ["SPEED_REQUEST", "GG_SCALE"] else '""'},
            "probability": ""
            }}''' for command in command_list]) + "\n"
            + "".join(lines[-3:]))
    elif prompt_version == 2:
        with open(os.path.join(os.path.dirname(__file__), 'prompt_template_2.txt'), 'r') as file:
            lines = file.readlines()
            header = lines[0:6]
            command_definitions = lines[6:23]
            footer = ("".join(lines[23:31]) + "["
            + ",\n".join([str([command_to_index[command], "null" if command not in ["SPEED_REQUEST", "GG_SCALE"] else "", 0]) for command in command_list])
            + "".join(lines[-3:]))
    elif prompt_version == 3:
        with open(os.path.join(os.path.dirname(__file__), 'prompt_template_3.txt'), 'r') as file:
            lines = file.readlines()
            header = lines[0:4]
            command_definitions = lines[4:21]
            footer = lines[21:]
    else:
        raise ValueError("Unsupported prompt version. Supported versions are 1, 2, and 3.")
    
    return "".join(header) + "".join([f"{command_definitions[command_to_index[command]-1]}" for command in command_list]) + "".join(footer)
    

    

def is_list_of_lists(obj):
    return isinstance(obj, list) and all(isinstance(item, list) for item in obj)

def parse_llm_output_to_list_of_lists(list_of_lists: str) -> list:
    try:
        cleaned = list_of_lists.replace("''", "0").replace('""', "0")
        cleaned = cleaned.replace("yes", "null").replace("no", "null")
        cleaned = cleaned.replace("None", "null")
        list_of_lists = json.loads(cleaned)
        if is_list_of_lists(list_of_lists):
            return list_of_lists
        else:
            raise ValueError("Parsed object is not a list of lists.")
    except:
        return []

def to_output_version_1(list_of_lists) -> str:
    try:
        list_of_lists = parse_llm_output_to_list_of_lists(list_of_lists)
        output = (
        """
        {
        """
            + ",\n".join([
            f'''    "{index_to_command[list_[0]]}": {{
            "value": {"null" if index_to_command[list_[0]] not in ["SPEED_REQUEST", "GG_SCALE"] else (0 if list_[1] == "" else int(list_[1]))},
            "probability": {0 if list_[2] == "" else float(list_[2])}
            }}''' for list_ in list_of_lists])
            + 
        """
        }
        """)
        return output
    except:
        return ""