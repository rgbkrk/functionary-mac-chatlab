from llama_cpp import Llama
from functionary.prompt_template import get_prompt_template_from_tokenizer
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from termcolor import colored
import json

import asyncio

# Model repository on the Hugging Face model hub
model_repo = "meetkai/functionary-7b-v2-GGUF"

# File to download
file_name = "functionary-7b-v2.f16.gguf"

# Download the file
local_file_path = hf_hub_download(repo_id=model_repo, filename=file_name)

# You can download gguf files from https://huggingface.co/meetkai/functionary-7b-v2-GGUF/tree/main
llm = Llama(model_path=local_file_path, n_ctx=4096, n_gpu_layers=-1)

# Create tokenizer from HF. 
# We found that the tokenizer from llama_cpp is not compatible with tokenizer from HF that we trained
# The reason might be we added new tokens to the original tokenizer
# So we will use tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("meetkai/functionary-7b-v2", legacy=True)
# prompt_template will be used for creating the prompt
prompt_template = get_prompt_template_from_tokenizer(tokenizer)

def run_inference(messages, tools):
    # Create the prompt to use for inference
    prompt_str = prompt_template.get_prompt_from_messages(messages + [{"role": "assistant"}], tools)
    token_ids = tokenizer.encode(prompt_str)

    gen_tokens = []
    # Get list of stop_tokens 
    stop_token_ids = [tokenizer.encode(token)[-1] for token in prompt_template.get_stop_tokens_for_generation()]
    # print("stop_token_ids: ", stop_token_ids)

    # We use function generate (instead of __call__) so we can pass in list of token_ids
    for token_id in llm.generate(token_ids, temp=0):
        if token_id in stop_token_ids:
            break
        gen_tokens.append(token_id)

    llm_output = tokenizer.decode(gen_tokens)

    # parse the message from llm_output
    response = prompt_template.parse_assistant_response(llm_output)

    return response


async def main():
    messages = [
        {"role": "user", "content": "what's the weather like in Santa Cruz, CA compared to Seattle, WA?"}
    ]

    from chatlab import tool_result, FunctionRegistry
    from pydantic import Field
    import random

    def get_current_weather(location: str = Field(description="The city and state, e.g., San Francisco, CA")):
        """Get the current weather"""

        return {
            "temperature": 75 + random.randint(-5, 5),
            "units": "F",
            "weather": random.choice(["sunny", "cloudy", "rainy", "windy"]),
        }

    fr = FunctionRegistry()
    fr.register(get_current_weather)

    print(colored("Tools: ", "cyan"))
    print(colored(json.dumps(fr.tools, indent=2), "cyan"))

    response = run_inference(messages, fr.tools)
    messages.append(response)

    if response.get('content') is not None:
        print(colored(f"Assistant: {response['content']}", "cyan"))

    if response.get('tool_calls') is not None:
        for tool in response['tool_calls']:
            requested_function = tool['function']
            result = await fr.call(requested_function['name'], requested_function['arguments'])
            print(colored(f"  𝑓  {requested_function['name']}({requested_function['arguments']})", "magenta"), " -> ", colored(str(result), "light_cyan"))

            tool_call_response = tool_result(tool['id'], content=str(result))
            # OpenAI does not require the name field, but it is required for functionary's tool_result
            tool_call_response['name'] = requested_function['name'] 

            messages.append(tool_call_response)
        
        # Run inference again after running tools
        response = run_inference(messages, fr.tools)
        print(colored(f"Assistant: {response['content']}", "yellow"))



if __name__ == "__main__":
    asyncio.run(main())