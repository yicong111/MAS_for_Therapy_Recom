import os
import json
import random
from openai import OpenAI
from pptree import *
import time 
import numpy as np
import dill
import jsonlines
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from mistralai import Mistral

def read_jsonlines(data_path):
    '''read data from jsonlines file'''
    data = []
    with jsonlines.open(data_path, "r") as f:
        for meta_data in f:
            data.append(meta_data)

    return data



class Agent:
    def __init__(self, instruction, role, model_info='Qwen2.5-72B-Instruct-GPTQ-Int4', img_path=None): #examplers=None,
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.messages = [
            {"role": "system", "content": instruction},
        ]
    def chat(self, message, img_path=None, chat_mode=True, max_retries=3): 
        retry_count = 0  
        success = False  
        model_name = "Qwen2.5-72B-Instruct-GPTQ-Int4"
        while retry_count < max_retries and not success:
            try:
                self.messages.append({"role": "user", "content": message})
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=4096,
                    temperature=0.6,
                )
                assistant_response = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": assistant_response})
                success = True  
                return assistant_response  
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")  
                time.sleep(2) 
        fallback_message = "Unable to generate a response at this time. Please try again later."
        print(f"Max retries reached. Returning fallback message.")
        self.messages.append({"role": "assistant", "content": fallback_message})
        return fallback_message



class Agent_DeepSeek: 
    def __init__(self, instruction, role, model_info='', img_path=None): 

        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        openai_api_key = ""
        openai_api_base = ""
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "OK."}
        ]
       
    def chat(self, message, img_path=None, chat_mode=True, max_retries=3): 
        retry_count = 0  
        success = False 
        model_name = self.model_info
        while retry_count < max_retries and not success:
            try:
                self.messages.append({"role": "user", "content": message})
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=4096,
                    temperature=0.6
                )
                assistant_response = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": assistant_response})
                success = True  
                return assistant_response 
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")  
                time.sleep(2)  
        fallback_message = "Unable to generate a response at this time. Please try again later."
        print(f"Max retries reached. Returning fallback message.")
        self.messages.append({"role": "assistant", "content": fallback_message})
        return fallback_message



class Agent_Mistral:
    def __init__(self, instruction, role, model_info='Mistral-Small-24B-Instruct-2501', img_path=None): 
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8002/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "OK."}
        ]
    def chat(self, message, img_path=None, chat_mode=True, max_retries=3): 
        """
        处理模型推理请求，添加异常处理和重试机制。
        """
        retry_count = 0  
        success = False  
        model_name = "Mistral-Small-24B-Instruct-2501"
        while retry_count < max_retries and not success:
            try:
                self.messages.append({"role": "user", "content": message})
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=4096,
                    temperature=0.6,
                )
                assistant_response = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": assistant_response})
                success = True  
                return assistant_response  
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}") 
                time.sleep(2) 
        fallback_message = "Unable to generate a response at this time. Please try again later."
        print(f"Max retries reached. Returning fallback message.")
        self.messages.append({"role": "assistant", "content": fallback_message})
        return fallback_message


class Agent_gpt: #chatgpt
    def __init__(self, instruction, role, model_info='gpt-4o', img_path=None): 

        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        openai_api_key = ""
        openai_api_base = "https://api.openai.com/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": "OK."}
        ]
       
    def chat(self, message, img_path=None, chat_mode=True, max_retries=3): 
        retry_count = 0  
        success = False  
        model_name = self.model_info

        while retry_count < max_retries and not success:
            try:
                self.messages.append({"role": "user", "content": message})
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    max_tokens=4096,
                    temperature=0.6
                )
                assistant_response = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": assistant_response})
                success = True 
                return assistant_response 

            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed: {str(e)}")  
                time.sleep(2) 
        fallback_message = "Unable to generate a response at this time. Please try again later."
        print(f"Max retries reached. Returning fallback message.")
        self.messages.append({"role": "assistant", "content": fallback_message})
        return fallback_message