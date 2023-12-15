from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import openai
import time
from utils.util_fnc import preprocess, generate_input_text


class LLM:
    def __init__(self, config):
        # Initialize the LLM object with a given configuration
        self.config = config
        
        # Load the appropriate model and tokenizer based on the configuration
        if 'lamma' in self.config['model']:
            # Initialize the Lamma2 tokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'], token=config['huggingface_token'])
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'], use_auth_token=True)

            # Check if CUDA is available and set the appropriate data type, If inference at GPU we need to set data type ro float16 (But when we inference at CPU, we need to set float32)
            if torch.cuda.is_available() and 'cuda' in self.config['device']:
                self.torch_dtype = torch.float16
            else:
                self.config['device'] = 'cpu'
                self.torch_dtype = torch.float32
                
            # Create a pipeline for the specified task
            self.pipeline = pipeline(
                'text-generation',
                model=self.config['model'],
                tokenizer=self.tokenizer,
                torch_dtype=self.torch_dtype,
                device_map='auto'
            )

        elif 'gpt' in self.config['model']:
            self._client = OpenAI()

        else:
            # Raise an error if an unsupported model is specified
            raise ValueError("Not a supported model")

    # Generate an answer based on the input x
    def answer(self, input_text_dict: dict):
        if 'lamma' in self.config['model']:
            
            # Preprocess the input and generate an answer using Lamma2
            
            x = ', '.join(f'{k}={v}' for k, v in input_text_dict.items())
            sequences = self.pipeline(x, **self.config['lamma_config'])
            return sequences[0]['generated_text'].replace(x, "")
        
        elif 'gpt' in self.config['model']:
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
                            {"role": "system", "content": input_text_dict['role_prompt']},
                            {"role": "assistant", "content": input_text_dict['step_prompt'] + 'For Example' + str(input_text_dict['chosen_example_prompt'])},
                            {"role": "user", "content": input_text_dict['text'] + '\n' + input_text_dict['fin_prompt']},
                        ]

            response = self._client.chat.completions.create(**self.config['openai_config'], messages=messages)
            return response.choices[0].message.content
        
        else:
            # Raise an error if an unsupported model is specified
            raise ValueError("Not a supported model")