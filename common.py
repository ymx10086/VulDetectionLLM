import ast
import logging
from fastchat.model import get_conversation_template

def get_init_msg(goal, target):
    return ""

def process_target_response(target_response, score, goal, target_str):
    return f"""
            LANGUAGE MODEL OUTPUT: {target_response}\n
        """ 

def conv_template(template_name):
    template = get_conversation_template(template_name)
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template