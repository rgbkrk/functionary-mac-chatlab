from llama_cpp import Llama
from functionary.prompt_template import get_prompt_template_from_tokenizer
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Model repository on the Hugging Face model hub
model_repo = "meetkai/functionary-7b-v2-GGUF"

# File to download
file_name = "functionary-7b-v2.f16.gguf"

# Download the file
local_file_path = hf_hub_download(repo_id=model_repo, filename=file_name)

# You can download gguf files from https://huggingface.co/meetkai/functionary-7b-v2-GGUF/tree/main
llm = Llama(model_path=local_file_path, n_ctx=4096, n_gpu_layers=-1)
messages = [
    {"role": "user", "content": "what's the weather like in Hanoi?"}
]

# Create tokenizer from HF. 
# We found that the tokenizer from llama_cpp is not compatible with tokenizer from HF that we trained
# The reason might be we added new tokens to the original tokenizer
# So we will use tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("meetkai/functionary-7b-v2", legacy=True)
# prompt_template will be used for creating the prompt
prompt_template = get_prompt_template_from_tokenizer(tokenizer)

# Before inference, we need to add an empty assistant (message without content or function_call)
messages.append({"role": "assistant"})


tools = [ # For functionary-7b-v2 we use "tools"; for functionary-7b-v1.4 we use "functions" = [{"name": "get_current_weather", "description":..., "parameters": ....}]
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Create the prompt to use for inference
prompt_str = prompt_template.get_prompt_from_messages(messages, tools)
token_ids = tokenizer.encode(prompt_str)

gen_tokens = []
# Get list of stop_tokens 
stop_token_ids = [tokenizer.encode(token)[-1] for token in prompt_template.get_stop_tokens_for_generation()]
print("stop_token_ids: ", stop_token_ids)

# We use function generate (instead of __call__) so we can pass in list of token_ids
for token_id in llm.generate(token_ids, temp=0):
    if token_id in stop_token_ids:
        break
    gen_tokens.append(token_id)

llm_output = tokenizer.decode(gen_tokens)

# parse the message from llm_output
result = prompt_template.parse_assistant_response(llm_output)
print(result)
