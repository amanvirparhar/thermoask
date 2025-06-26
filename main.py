import ollama

model = "qwen2.5:7b"
default_temp = 0.6

prompt = """Solve the following math expression (show your work): "1 + 5 * 3 - 4 / 2".
Then, write a really abstract poem that contains the answer to this expression."""

system_prompt = """You have access to thermoask_tool which adjusts your sampling temperature for optimal performance. Call it before each distinct type of task within your response. Follow this pattern:

1. Call thermoask_tool() for the first task, specifying new temperature for the task
2. Immediately generate content for the first task using the modified temperature
3. Call thermoask_tool() for the next task, specifying new temperature for the task
4. Immediately generate content for that task using the modified temperature
5. Repeat, if necessary

IMPORTANT: Always generate content immediately after each tool call, don't call multiple tools in a row.

Temperature guide:
- 0.0-0.3: Precise tasks like solving math problems, coding
- 0.4-0.9: Balanced tasks like non-fiction summaries, essays
- 1.0-2.0: Creative and artistic tasks like fictional storytelling, brainstorming
- 2.0+: Truly random tasks like random number generation, unpredictable outputs

Use thermoask_tool(task_description, reasoning_space, temperature) where you specify:
- task_description: what task you're currently dealing with
- reasoning_space: which temperature category you think the task falls under, why, and what final temperature you are leaning towards (aim for at least 2 sentences of explanation)
- temperature: the final temperature value you want to switch to"""

print("generating with dynamic temperature adjustment...")


def thermoask_tool(task_description: str, reasoning_space: str, temperature: float) -> str:
    print(
        f"[tool called: '{task_description}' -> temp {temperature}; reasoning: '{reasoning_space}']")
    return f"Temperature was successfully set to {temperature} for '{task_description}'"


def run_conversation():
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]

    current_temp = default_temp

    while True:
        response = ollama.chat(model,
                               messages=messages,
                               tools=[thermoask_tool],
                               options={'temperature': current_temp})

        print(response['message']['content'])

        messages.append({
            'role': 'assistant',
            'content': response['message']['content'],
            'tool_calls': response['message'].get('tool_calls')
        })

        if response['message'].get('tool_calls'):
            for tool_call in response['message']['tool_calls']:
                if tool_call['function']['name'] == 'thermoask_tool':
                    args = tool_call['function']['arguments']
                    current_temp = args['temperature']

                    result = thermoask_tool(**args)
                    print(f"\n[temperature adjusted to {current_temp}]\n")

                    messages.append({
                        'role': 'tool',
                        'content': result,
                        'tool_call_id': tool_call.get('id', 'temp_adjust')
                    })

                    response = ollama.chat(model,
                                           messages=messages,
                                           tools=[thermoask_tool],
                                           options={'temperature': current_temp})

                    print(response['message']['content'])

                    messages.append({
                        'role': 'assistant',
                        'content': response['message']['content']
                    })

                    current_temp = default_temp
                    print(f"\n[temperature reset to default {default_temp}]\n")
        else:
            break


run_conversation()
