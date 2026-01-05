import os
import json
from typing import List,Dict
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt

from openai import OpenAI
import random

__registered_evaluators__ = {}

def register_evaluator(cls):
    """
    Decorator function to register classes with the registered_evaluators list.
    """
    __registered_evaluators__[cls.__name__] = cls
    return cls

def get_evaluator_cls(clsname):
    """
    Return the evaluator class with the given name.
    """
    try:
        return __registered_evaluators__.get(clsname)
    except:
        raise ModuleNotFoundError('Cannot find evaluator class {}'.format(clsname))


class OpenaiPoolRequest:
    def __init__(self, pool_json_file=None):
        self.pool:List[Dict] = []
        self.now_pos = -1
        __pool_file = pool_json_file
        if os.environ.get('API_POOL_FILE',None) is not None:
            __pool_file = os.environ.get('API_POOL_FILE')
        if os.path.exists(__pool_file):
            self.pool = json.load(open(__pool_file))
            self.now_pos = random.randint(-1, len(self.pool))
        # print(__pool_file)
        # Support standard env vars
        api_key = os.environ.get('OPENAI_KEY') or os.environ.get('OPENAI_API_KEY')
        api_base = os.environ.get('OPENAI_API_BASE') or os.environ.get('OPENAI_BASE_URL')
        
        if api_key is not None:
            self.pool.append({
                'api_key': api_key,
                'api_base': api_base,
                'organization': os.environ.get('OPENAI_ORG', None),
                'api_type': os.environ.get('OPENAI_TYPE', None),
                'api_version': os.environ.get('OPENAI_VER', None)
            })

    def request(self,messages,**kwargs):
        # Use pool config and force base/model to avoid 400s from missing fields.
        if not self.pool:
            raise RuntimeError("API pool is empty; please provide api_pool with api_key/api_base/model.")
        self.now_pos = (self.now_pos + 1) % len(self.pool)
        key_pos = self.now_pos
        item = self.pool[key_pos]
        api_key = item.get('api_key')
        api_base = item.get('api_base', None)
        model = item.get('model', None)
        # Remove stale kwargs entries.
        kwargs.pop('api_key', None)
        kwargs.pop('api_base', None)
        # Force model selection.
        if model:
            kwargs['model'] = model
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.chat.completions.create(messages=messages, **kwargs)
        return response
    
    def __call__(self,messages,**kwargs):
        return self.request(messages,**kwargs)
   
