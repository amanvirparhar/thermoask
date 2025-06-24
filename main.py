import ollama

model = "qwen2.5:7b"

prompt = """Solve the following math expression: "1 + 5 * 3 - 4 / 2".
Then, write a really abstract poem that contains the answer to this expression."""

system_prompt = """You have access to thermoask_tool which adjusts your temperature for optimal performance. Call it before each distinct type of task within your response.

Temperature guide with approximate ranges:
- 0.0-0.3: Precise tasks like math, coding
- 0.4-0.8: Balanced and factual tasks like explanations, summaries
- 0.9-2.0: Creative and artistic tasks like storytelling, brainstorming
- 2.0+: Truly random tasks like random number generation, unpredictable outputs

Use thermoask_tool(task_description, reasoning_space, temperature) where you specify both what task you're deaing with and what temperature you want to switch to."""

print("generating with dynamic temperature adjustment...")


def thermoask_tool(task_description: str, temperature: float) -> str:
    print(f"[tool called: '{task_description}' -> temp {temperature}]")
    return f"Temperature set to {temperature}"


def run_conversation():
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]

    current_temp = 1.0

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
        else:
            break


run_conversation()
