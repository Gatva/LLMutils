import openai
import os
import torch
import logging
import time
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

logging.basicConfig(filemode='w', filename='log', format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO)


class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        pass

    def generate(self, prompt):
        raise NotImplementedError("")


class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 cache_dir=None,
                 device='cuda',
                 quantization=False,
                 system_prompt=None):
        super().__init__()
        self.nf4_config = None
        if quantization:
            self.nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            quantization_config=self.nf4_config,
            device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir)
        self.device = device
        self.system_prompt = system_prompt
        self.model_path=model_path


    def set_system_prompt(self):
        if self.system_prompt is None and 'Llama-2' in self.model_path:
            self.systemp_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
                                  "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
                                  "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
                                  "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
                                  "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_prompt = None
    def generate(self, prompt, temperature=1.0, max_tokens=512, n=1, max_trial=5, fail_sleep_time=5):

        message = [
            {'role': 'user', 'content': prompt}
        ]
        self.set_system_prompt()
        if self.system_prompt:
            message = [
                {'role': 'user', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt}
            ]
        text = self.tokenizer.apply_chat_template(message, tokenize=False)
        input = self.tokenizer(text, return_tensors="pt").to(self.device)

        for _ in range(max_trial):
            try:
                responses = self.model.generate(
                    **input,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                return [self.tokenizer.decode(responses[0], skip_special_tokens=True)]
            except:
                logging.exception("model failure")
                time.sleep(fail_sleep_time)

        return ["" for _ in range(n)]


class OpenAILLM(LLM):
    def __init__(self, model_path='gpt-3.5-turbo', api_key=os.getenv("OPENAI_API_KEY")):
        super().__init__()
        self.model = model_path
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt, temperature=1.0, max_tokens=512, n=1, max_trial=5, fail_sleep_time=5):
        massage = [
            {'role': 'user', 'content': prompt}
        ]
        for _ in range(max_trial):
            try:
                responses = self.client.chat.completions.create(model=self.model,
                                                                messages=massage,
                                                                temperature=temperature,
                                                                n=n,
                                                                max_tokens=max_tokens)
                return [responses.choices[i].message.content for i in range(n)]
            except:
                logging.exception(
                    "There seems to be something wrong with your ChatGPT API. Please follow our demonstration in the slide to get a correct one.")
                time.sleep(fail_sleep_time)

        return ["" for i in range(n)]


class GeminiLLM(LLM):
    def __init__(self, model_path='gemini-pro', api_key=os.getenv("Gemini_API_KEY")):
        super().__init__()
        self.model = model_path
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt, temperature=1.0, max_tokens=512, n=1, max_trial=5, fail_sleep_time=5):
        content = [
            {'role': 'user', 'parts': prompt}
        ]
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        for _ in range(max_trial):
            try:
                responses = self.client.generate_content(
                    contents=content,
                    generation_config=generation_config,
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE", },
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE", },
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE", },
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE", }]
                )
                return [responses.text]
            except:
                logging.exception(
                    "There seems to be something wrong with your Gemini API. Please follow our demonstration in the slide to get a correct one.")
                time.sleep(fail_sleep_time)
        return ["" for i in range(n)]
