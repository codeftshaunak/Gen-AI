# step-1 import requirement
from dotenv import load_dotenv
import json
load_dotenv()
from openai import OpenAI
client = OpenAI()


# step-2 write the system prompt
SYSTEM_PROMPT = """
You're a trained ai model on mathematics, great problem solver to help user with math knowledge. You're always follow each step. 

You're always followed rules given bellow 
1. User gives you an input. 
2. You analyse the input. 
3. You start thinking about the input. 
4. You generate the answer.
5. You validate the answer again. 
6. The you send the result as output. 

Following steps are mostly like
Input -> Analyse -> Think -> Output -> Validate -> Result. 

Few strict instruction:
1. You never skip the steps. 
2. Always perform one step at a time. 
3. Output in json format always. 
4. Follow the rules given strictly. 
5. Analyse the user query very carefully. 

Output example 
{{"step":"<string>", content:"<string>"}}

Example 
Input: What is 4-3 ? 
Output: {{'step':'analysis','content':'User asking about what is the value of 4 - 3'}}
Output: {{'step':'think','content':'For solving this problem i need ot follow rule of BODMAS etc and solve this'}}
Output: {{'step':'output','content':'1'}}
Output: {{'step':'validate','content':'Seems like we are getting the 1 value which is seems correct'}}
Output: {{'step':'result','content':'Final answer is 1'}}
"""


# Need to generate a message 
messages = [
    {'role':'system', 'content':SYSTEM_PROMPT},
]

previous_steps=[]


# Query from input & append message
query = input("What is your question:")
messages.append({'role':'user','content':query})


def call_for_thinking ():
    context_message =[{ 'role':'system', 'content':"You're here to thinking the previous step and make sure it follow the proper rules for ex: user ask about 7-3*2 in this your output will be more like For solving this problem i need ot follow rule of BODMAS etc and solve this. Apart from that your job is to make sure that if user put anything incomplete you need predict it and fix it for resolve. For example if user put an incomplete question or missing parentheses such as 5-6 ( the last operator is given by mistake. That means user either put it by mistake or he want to add some value. As he is not added any value we can predict that as 0. But at any cost need to complete the calculation. As a thinking model you need to predict what user want. "}]
    context_content= "Here is all the previous messages\n"

    for steps in previous_steps:
        context_content+= f"Step '{steps['step']}:{steps['content']} \n"
        
    context_content+= "\n Provide detail thinking based on the previous steps."
    context_message.append({'role':'user', 'content':context_content})

    response = client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=context_message,
    )
    print(f"\n🧠 THINKING AGENT: {response.choices[0].message.content}\n")

    return response


# Let's make it step by step
while True:
    # OpenAI request & call 
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        response_format={"type": "json_object"},
        messages=messages,
    )

    messages.append({"role":"assistant","content":response.choices[0].message.content})
    parsed_response = json.loads(response.choices[0].message.content)  

    if parsed_response['step'] == 'think':
        previous_steps.append(parsed_response)
        thinking_result = call_for_thinking()
        messages.append({"role": "system", "content": f"Thinking Agent Analysis: {thinking_result}"})


    if parsed_response['step'] != 'result':
        print(f'\n BOT({parsed_response['step']}:{parsed_response['content']}) \n')
        previous_steps.append(parsed_response)
        continue

    if parsed_response['step'] == 'result':
        print(f'\n 👑 ({parsed_response['step']}:{parsed_response['content']})')
        break