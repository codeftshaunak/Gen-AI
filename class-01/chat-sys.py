import json
from pyexpat.errors import messages
from dotenv import load_dotenv;
load_dotenv()

from openai import OpenAI;
client=OpenAI()

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

# response = client.chat.completions.create(
#     model="gpt-4.1-mini",
#     response_format={"type": "json_object"},
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": "What is 4*4?"} ,
#         {"role": "assistant", "content": json.dumps({"step":"analyze", "content":"The user is asking for the result of 4 multiplied by 4, which is a basic multiplication problem."})},
#         {"role": "assistant", "content": json.dumps({"step": "think", "content": "To solve 4 multiplied by 4, I will multiply the two numbers: 4 times 4 equals 16."} )},
#         {"role": "assistant", "content": json.dumps({"step": "output", "content": "16"} )},
#         {"role": "assistant", "content": json.dumps({"step": "validate", "content": "I need to ensure the output is correct. Multiplying 4 by 4 results in 16, so the output is correct."} )},
#         {"role": "assistant", "content": json.dumps({"step": "result", "content": "The final result of 4 multiplied by 4 is 16."}  )},
#     ]
# )

# print("\n\nBOT:", response.choices[0].message.content,"\n\n")

messages=[{"role":"system","content":SYSTEM_PROMPT}]

query = input("Enter your query: ")
messages.append({"role":"user","content":query})
# messages.append({"role":"system","content":SYSTEM_PROMPT})

while True:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=messages
    )

    messages.append({"role":"assistant","content":response.choices[0].message.content})
    parsed_response = json.loads(response.choices[0].message.content)

    if parsed_response['step'].lower()!="result":
        print(f"\n\nBOT ({parsed_response['step']}): {parsed_response['content']}\n\n")
        continue
    print(f"\n\nBOT ({parsed_response['step']}): {parsed_response['content']}\n\n")
    break
