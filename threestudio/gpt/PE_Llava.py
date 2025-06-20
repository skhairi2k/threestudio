import openai
import json, os, pdb, time, sys, json
import pickle
import pandas as pd
import re
import base64
import requests

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from threestudio.utils.misc import cleanup

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 


def get_response_with_img(msg, image_path):
    
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)
    
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here is an image"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            },
            },
            {"type": "text", "text": msg},
        ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    
    # Generate
    ouput = model.generate(**inputs, max_new_tokens=300)
    return processor.decode(ouput[0], skip_special_tokens=True)

def get_response(msg):
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a helpful assistant!"},
            {"type": "text", "text": msg},
        ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt"
    
    ).to("cuda:0", torch.float16)

    # Generate
    model.to("cuda:0")
    generate_ids = model.generate(**inputs, max_new_tokens=300)
    model.cpu()
    cleanup()
    return processor.batch_decode(generate_ids, skip_special_tokens=True)
    

def ImageEvaluate(attr, part, image_pth):

    ques = "Look at this figure, and if the " + part + " is " + attr + ", please say 'YES' only. If the " + part + " is not " + attr + ", please say 'NO' only. Don't say anything else."

    res = get_response_with_img(ques, image_pth)

    print(f"res: {res}")

    return res

from textwrap import dedent

SYSTEM_HEADER = dedent("""
    ### System
    You are an expert scene decomposition assistant for text-to-3D generation.
""").strip()

DEFINITIONS = dedent("""
    ### Definitions
    - body: **main entity** performing the action (singular).
    - instance: an **attachable object** (hat, shirt, etc.).
    - attribute: the **adjective phrase** that modifies an instance (blue, furry…).
    - stratification: The first layer represents clothes that are the **CLOSEST** to the skin and the last layer clothes that are the **FARTHEST** from the skin (inside-to-outside layering). Use token EXTEND to mark a new layer. 
""").strip()

TASK = dedent("""
    ### Task (follow exactly)
    1. Extract `body`, `instances` (dict instance → attribute).
    2. Determine `stratification_order` (list of strings; put EXTEND between layers).
    3. Compose `sub_prompts`:
       - `sub_prompts[i]` describes all layers **up to** i (inclusive).
    4. Return **only** valid JSON that fits the schema below.
""").strip()

SCHEMA = dedent("""
    ### Output Schema
    ```json
    {
      "body": "<string>",
      "instances": { "<instance>": "<attribute>", "...": "..." },
      "stratification_order": ["<instance attr>", "EXTEND", "..."],
      "sub_prompts": ["<prompt L1>", "<prompt L2>", "..."]
    }
    ```
""").strip()

EXAMPLE = dedent("""
    ### Few-shot Examples
    #### Input
    A yellow dog wearing a blue shirt, a black hat and a red coat is barking.
    #### Output
    ```json
    {
      "body": "yellow dog",
      "instances": { "shirt": "blue", "hat": "black", "coat": "red" },
      "stratification_order": ["blue shirt", "black hat", "EXTEND", "red coat"],
      "sub_prompts": [
        "A yellow dog wearing a blue shirt and a black hat is barking.",
        "A yellow dog wearing a red **coat**, shirt and black hat is barking."
      ]
    }
    #### Input
    A man wearing red shirt, striped scarf, blue trousers, leather shoes and denim jacket.
    #### Output
    ```json
    {
      "body": "a man",
      "instances": { "shirt": "red", "scarf": "striped", "trousers": "blue", "shoes": "leather", "jacket": "denim" },
      "stratification_order": ["red shirt", "blue trousers", "leather shoes", "EXTEND" "striped scarf", "denim jacket"],
      "sub_prompts": [
        "A man wearing red shirt, blue trousers and leather shoes.",
        "A man wearing a striped **scarf**, denim **jacket**, shirt, trousers and leather shoes."
      ]
    }              
    ### Input
    A clown with red nose and white face, wears red jacket, red trousers, yellow-green shirt, black shoes and green wig.
    #### Output
    ```json
    {
        "body": "a clown with red nose and white face",
        "instances": {
          "wig": "green",
          "shirt": "yellow-green",
          "jacket": "red",
          "trousers": "red",
          "shoes": "black"
        },
        "stratification_order": [
          "yellow-green shirt",
          "red trousers",
          "black shoes",
          "green wig",
          "EXTEND",
          "red jacket"
        ],
        "sub_prompts": [
          "a clown with red nose and white face, wears yellow-green shirt  red trousers and black shoes green wig ", (notice how jacket is not included)
          "a clown with red nose and white face, wears red **jacket**, shirt, trousers, black shoes and green wig"
        ]
      }
    ```
""").strip()

def make_prompt(user_prompt: str) -> str:
    """Compose the full prompt string for the LLM."""
    return "\n\n".join([
        SYSTEM_HEADER,
        DEFINITIONS,
        TASK,
        SCHEMA,
        EXAMPLE,
        "### Input\n" + user_prompt.strip()
    ])

def get_chain(prompt):
    # ques = 'I am now going to use a model to generate text to 3D. This generation is done "inside-to-out", so I need to split the whole sentence into inside-out order as well. Specifically, a sentence contains a subject, instances, and attributes. For example, a yellow dog wearing a blue shirt, a black hat and a red coat is barking. The subject is the yellow dog is barkings, and the instances include shirt, hats, and coat, with the corresponding attributes blue, black, and red. What you need to do is, given a prompt, extract the body and the corresponding instance to the attribute. The next step is to carry out stratification. The rule of stratification is that from inside to outside, if there is a relatively obvious occlusion relationship between instances, then it is necessary to expand one layer down. For example, in the above example, the first layer is shirt and hats. The second layer is the coat. So the stratification order is (blue shirt, black hat, EXTEND, red coat) in which the EXTEND means for the next layer. And the sub-prompt of the first layer is: "A yellow dog wearing a blue shirt and a black hat is barking". And the sub-prompt of the second layer is: "A yellow dog wearing a red coat is barking." So what you end up returning is to tell me the body, the instances with corresponding attributes the stratification order and the sub-prompts of each layer. The prompt is: ' + prompt
    # There are some diffenrences when using SD3
    # ques = 'I am now going to use a model to generate text to 3D. This generation is done "inside-to-out", so I need to split the whole sentence into inside-out order as well. Specifically, a sentence contains a subject, instances, and attributes. For example, a yellow dog wearing a blue shirt, a black hat and a red coat is barking. The subject is the yellow dog is barkings, and the instances include shirt, hats, and coat, with the corresponding attributes blue, black, and red. What you need to do is, given a prompt, extract the body and the corresponding instance to the attribute. The next step is to carry out stratification. The rule of stratification is that from inside to outside, if there is a relatively obvious occlusion relationship between instances, then it is necessary to expand one layer down. For example, in the above example, the first layer is shirt and hats. The second layer is the coat. So the stratification order is (blue shirt, black hat, EXTEND, red coat) in which the EXTEND means for the next layer. And the sub-prompt of the first layer is: "A yellow dog wearing a blue shirt and a black hat is barking". And the sub-prompt of the second layer is: "A yellow dog wearing a red coat, shirt and black hat is barking." (Notice that the later layer contains the previous parts, and contains the previous attributes whose parts are not heavily obscured but not contains the previous attributes whose parts are severely obscured. For example, for here, coat occludes most of the shirt but little hat, so the shirt has no attribute but the hat has black attribute in this layer.) So what you end up returning is to tell me the body, the instances with corresponding attributes the stratification order and the sub-prompts of each layer. The prompt is: ' + prompt
    # ques += '\n'
    # ques += 'NOTE: please only output a set format of json file. The keys are "body", "instances", "stratification_order", "sub_prompts". And the value of "body" is string, the value of "instances" is a dict, the value of "stratification_order" is a list of strings, and the value of "sub_prompts" is a list of strings.'
    
    # ques = make_prompt(prompt)
    # res = get_response(ques)
    
    # print(f"the type of res is {type(res)}, res: {res}")
    import json
    # idx = 0
    # idx_2 = 0
    # for i in range(len(res)):
    #     if res[i] == "`":
    #         idx = i
    #         break
    # for i in range(idx+5, len(res)):
    #     if res[i] == "`":
    #         idx_2 = i+2
    #         break
    # res = res[idx:idx_2]
    # res = res.strip("```json").strip("```")
    
    # json_dict = json.loads(res)
    # json_dict = json.loads(res[0].split('json')[-1].strip("``` "))

    json_dict = json.loads(
    """
    {
        "body": "a man",
        "instances": {
          "hat": "green",
          "shirt": "yellow",
          "coat": "black",
          "trousers": "pink",
          "shoes": "blue"
        },
        "stratification_order": [
          "yellow shirt",
          "pink trousers",
          "blue shoes",
          "green hat",
          "EXTEND",
          "black coat"
        ],
        "sub_prompts": [
          "a man in yellow shirt inside, green hat, blue shoes, and pink trousers is waving", 
          "a man in black coat, shirt inside, hat, shoes, and trousers is waving"
        ]
      }
    """)



    
    ### an example
    # json_dict = {
    #     "body": "a clown with red nose and white face",
    #     "instances": {
    #       "wig": "green",
    #       "shirt": "yellow-green",
    #       "jacket": "red",
    #       "trousers": "red",
    #       "shoes": "black"
    #     },
    #     "stratification_order": [
    #       "yellow-green shirt",
    #       "red trousers",
    #       "black shoes",
    #       "green wig",
    #       "EXTEND",
    #       "red jacket"
    #     ],
    #     "sub_prompts": [
    #       "a clown with red nose and white face, wears green wig, red trousers, black shoes and yellow-green shirt",
    #       "a clown with red nose and white face, wears red jacket trousers, shirt, shoes and green wig"
    #     ]
    #   }
    return json_dict


if __name__ == "__main__":
    res = get_chain("A man in black coat, yellow shirt, pink trousers, blue shoes and green hat is waving")
    print(res)