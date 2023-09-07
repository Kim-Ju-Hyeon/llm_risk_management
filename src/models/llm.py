from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import openai
import time
from utils.util_fnc import cleaning_text, generate_input_text


class LLM:
    def __init__(self, config):
        # Initialize the LLM object with a given configuration
        self.config = config
        
        # Load the appropriate model and tokenizer based on the configuration
        if self.config['model'] == 'lamma2':
            # Initialize the Lamma2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'], token=config['huggingface_token'])

            # Check if CUDA is available and set the appropriate data type, If inference at GPU we need to set data type ro float16 (But when we inference at CPU, we need to set float32)
            if torch.cuda.is_available() and 'cuda' in config['device']:
                self.torch_dtype = torch.float16
            else:
                config['device'] = 'cpu'
                self.torch_dtype = torch.float32
                
            # Create a pipeline for the specified task
            self.pipeline = pipeline(
                config['task'],
                model=config['model_name'],
                tokenizer=self.tokenizer,
                torch_dtype=self.torch_dtype,
                device=config['device']
            )
        elif self.config['model'] == 'gpt3.5':
            # we use GPT-3.5 at Open AI API so we don't need any extra pipeline
            pass
        else:
            # Raise an error if an unsupported model is specified
            raise ValueError("Not a supported model")

    # Generate an answer based on the input x
    def answer(self, x):
        if self.config['model'] == 'lamma2':
            # Preprocess the input and generate an answer using Lamma2
            x = '\n'.join(x)
            sequences = self.pipeline(x, **self.config['lamma_config'])
            return sequences[0]['generated_text'].replace(x, "")
        elif self.config['model'] == 'gpt3.5':
            '''
            Preprocess the input and generate an answer using GPT-3.5
            There is an three role in chatGPT Open AI API
            1) system - GPT의 역활이나 어떤식으로 우리에 답변을 해줘야되는지에 대한 지시사항을 줄 수 있습니다.
            2) assistant - GPT의 답변, history, 알고잇어야되는 정보 등등이 될 수 있습니다
            3) user - 우리가 GPT에게 물어보는 질문
            so I made the prompt separately role, step, example, last prompt
            role 프롬프트는 LLM에게 구체적인 역활을 부여해서 답변 퀄리티를 높일수 있습니다.
            step prompt는 LLM에게 어떤 단계로 답변을 해야될지를 명시해주며 이 또한 답변 퀄리티를 높입니다.
            example prompt는 실제 내가 어떤 답변을 원하는지에 대한 구체적인 예시를 주어 우리가 원하는 구체적 답변을 얻을 수 있게 해주는 prompt입니다.
            last prompt는 마지막으로 지시사항을 다시한번 상기시켜주는 prompt 입니다.

            role, step, example prompt는 원하는 답변의 요구사항으로서 -> system
            article LLM이 알고있어야되는 정보이므로 -> assistant
            last prompt는 마지막으로 저희가 질문을 하는 부분으로 -> user
            로 message를 구성하였습니다.
            '''
            messages = [
                {"role": "system", "content": '\n'.join(x[:3])},
                {"role": "assistant", "content": x[3]},
                {"role": "user", "content": x[4]}
            ]
            response = openai.ChatCompletion.create(**self.config['openai_config'], messages=messages)
            return response['choices'][0]['text']
        else:
            # Raise an error if an unsupported model is specified
            raise ValueError("Not a supported model")

    def evaluate_text(self, article, prompt_dict):
        # Initialize variables for time tracking
        start_time = time.time()
        elapsed_time = 0

        # Loop to generate and evaluate text until the time limit is reached
        while elapsed_time < self.config.time_limit:
            # Generate the input text
            input_text = generate_input_text(prompt_dict, article)
            # Get the generated answer
            gen_text = self.answer(input_text)
            # Clean the generated text
            cleaned_output = cleaning_text(gen_text)
            # Skip the output if it's too long
            if len(cleaned_output) >= 500:
                continue

            # Evaluate the cleaned output
            if "True" in cleaned_output and "False" not in cleaned_output:
                return True, cleaned_output
            elif "False" in cleaned_output and "True" not in cleaned_output:
                return False, cleaned_output

            # Update elapsed time
            elapsed_time = time.time() - start_time

        # If time limit is reached without a definitive answer
        return None, None
