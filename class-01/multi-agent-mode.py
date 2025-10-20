from dotenv import load_dotenv
import json
import os
load_dotenv()
from openai import OpenAI
client = OpenAI()

SYSTEM_PROMPT="""
You're an helpful ai assistant who is specialized in resolving users queries. 
For the given user input, analyze the input and break down the problem step by step. 

The steps are as follows: 
1. User give you an input. 
2. You analyze the input
3. You start thinking
4. Think for several times
5. Then return to a outcome. 

Follow the step as a sequence: 
1. Analyze
2. Think
3. Output 
4. Validate
5. Result 

Rules: 
1. Follow the steps strictly.
2. Do not skip any step.
3. Follow the strict json output format.
4. Always perform one step at a time & wait for the next input.
5. Carefully analyze the user query. 

Output Format:
{{"step":"<string>", "content":"<string>"}}

Example: 
Input: "What is 2+4?"
Output:{{"step":"analyze", "content":"The user is asking for a basic addition."}}
Output:{{"step":"think", "content":"To this addition problem, I must start from left to right."}}
Output:{{"step":"output", "content":"6"}}
Output:{{"step":"validate", "content":"I need to ensure the output is correct. It seems to be correct."}}
Output:{{"step":"result", "content":"The final result is 6."}}

"""


messages=[
    {"role":"system","content":SYSTEM_PROMPT},
]

query = input("Enter your football query: ")
messages.append({"role":"user","content":query})

previous_steps=[]

# Function to call others modal for think steps only
def call_for_thinking():
    # Build context from all previous steps
    context_messages = [
        {"role": "system", "content": "You are a specialized thinking assistant for football queries. Analyze the previous steps and provide deep reasoning."}
    ]

    # Create context from previous steps
    context_content = "Previous analysis steps:\n"
    for step in previous_steps:
        context_content += f"Step '{step['step']}': {step['content']}\n"

    context_content += "\nProvide detailed thinking and reasoning based on this context."
    context_messages.append({"role": "user", "content": context_content})

    # Call GPT-3.5-turbo for deep thinking
    thinking_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=context_messages,
        max_tokens=500,
        temperature=0.7
    )

    thinking_result = thinking_response.choices[0].message.content
    print(f"\n🧠 THINKING AGENT: {thinking_result}\n")

    return thinking_result


while True:
    response = client.chat.completions.create(
    model="gpt-4.1-mini",
    response_format={"type": "json_object"},
    messages=messages
    )

    messages.append({"role":"assistant","content":response.choices[0].message.content})
    parsed_response = json.loads(response.choices[0].message.content)    

    if parsed_response['step'].lower()=="think":
        previous_steps.append(parsed_response)
        thinking_result = call_for_thinking()
        # Add thinking result as context for the main model
        messages.append({"role": "system", "content": f"Thinking Agent Analysis: {thinking_result}"})
        

    if parsed_response['step'].lower() != "result":
        print(f'\nBOT({parsed_response['step']}):{parsed_response['content']}\n')
        previous_steps.append(parsed_response)
        continue

    if parsed_response['step'].lower()=='result':
        print(f'\nBOT({parsed_response['step']}):{parsed_response['content']}\n')
        break

