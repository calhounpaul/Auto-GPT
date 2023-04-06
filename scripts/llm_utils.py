import openai
from config import Config
cfg = Config()
import requests, json
import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.openai_deployment_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif cfg.use_fc_vicuna:
        #https://github.com/lm-sys/FastChat
        model_name = "vicuna-13b"
        controller_addr = "http://localhost:21001"
        ret = requests.post(controller_addr + "/get_worker_address",json={"model": model_name})
        worker_addr = ret.json()["address"]
        conv = conv_v1_2.copy()
        for message in messages:
            if message["role"] == "user":
                conv.append_message(conv.roles[0], message["content"])
            else:
                conv.append_message(conv.roles[1], message["content"])
        prompt = conv.get_prompt()
        pload = {
            "model": model_name,
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "stop": conv.sep,
        }
        headers = {"User-Agent": "fastchat Client"}
        response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True)
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                final_out = data["text"].split(conv.sep)[-1]
        if final_out.startswith(conv.roles[1]):
            final_out = final_out[len(conv.roles[1])+1:].strip()
        final_out = final_out.split(conv.roles[0]+": ")[0].split(conv.roles[1]+": ")[0]
        return final_out
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return response.choices[0].message["content"]
