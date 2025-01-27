import os
from anthropic import Anthropic, AnthropicBedrock, HUMAN_PROMPT, AI_PROMPT
from negotiationarena.agents.agents import Agent
import time
from copy import copy, deepcopy
from negotiationarena.constants import AGENT_TWO, AGENT_ONE


class ClaudeAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        model: str = "claude-2.1",
        use_system_prompt=True,
        api_type: str = "anthropic"
    ):
        super().__init__(agent_name)
        self.run_epoch_time_ms = str(round(time.time() * 1000))

        self.conversation = []
        self.model = model
        self.use_system_prompt = use_system_prompt
        self.api_type = api_type
        if self.api_type == "anthropic":
            self.role_to_prompt = {
                "user": HUMAN_PROMPT,
                "assistant": AI_PROMPT,
            }
        elif self.api_type == "amazon_bedrock":
            self.role_to_prompt = {
                "user": "user",
                "assistant": "assistant",
            }
        else:
            raise ValueError(
                "api_type must be either 'anthropic' or 'amazon_bedrock'"
            )
        self.prompt_entity_initializer = "system"
        if self.api_type == "anthropic":
            self.anthropic = Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=os.environ.get("ANTHROPIC_API_KEY"),      
            )
        elif self.api_type == "amazon_bedrock":
            self.anthropic = AnthropicBedrock(
                aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_region=os.environ.get("AWS_DEFAULT_REGION")
            )     

    def init_agent(self, system_prompt, role):
        if AGENT_ONE in self.agent_name:
            # we use the user role to tell the assistant that it has to start.
            self.update_conversation_tracking(
                self.prompt_entity_initializer, system_prompt
            )
            self.update_conversation_tracking("user", role)

        elif AGENT_TWO in self.agent_name:
            system_prompt = system_prompt + role
            self.update_conversation_tracking(
                self.prompt_entity_initializer, system_prompt
            )
        else:
            raise "No Player 1 or Player 2 in role"

        # system_prompt = system_prompt + role
        # self.update_conversation_tracking(self.prompt_entity_initializer, system_prompt)

    def __deepcopy__(self, memo):
        """
        Deepcopy is needed because we cannot pickle the anthropic object.
        :param memo:
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if (type(v) == Anthropic): 
                v = "AnthropicObject"
            if (type(v) == AnthropicBedrock):
                v = "AnthropicBedrockObject"
            setattr(result, k, deepcopy(v, memo))
        return result

    def messages_to_prompt(self, messages):
        """
        We convert the messages into antrhopic format. note that we can have the system prompt on top or not. If
        we are not using the system prompt, we are going to add it to the first HUMAN message.
        :param messages:
        :return:
        """
        prompt = ""

        # if we use the claude2.1 system prompt style. we add it on top without any role.
        if self.use_system_prompt:
            text = messages[0]["content"]
            prompt += f"{text}"

            for message in messages[1:]:
                role_prompt = self.role_to_prompt[message["role"]]
                prompt += f"{role_prompt} {message['content']}"

        else:
            text = messages[0]["content"]
            prompt += f"{self.role_to_prompt['user']} {text}"

            text = messages[1]["content"]
            prompt += f"\n\n{text}"

            for message in messages[2:]:
                role_prompt = self.role_to_prompt[message["role"]]
                prompt += f"{role_prompt} {message['content']}"

        return prompt + f"\n\n{self.role_to_prompt['assistant']}"

    def messages_to_anthropic_format(self, messages):

        anthropic_messages = []
        if self.use_system_prompt:
            system_prompt = self.conversation[0]["content"]

            for message in messages[1:]:
                anthropic_messages.append({
                    "role": self.role_to_prompt[message["role"]],
                    "content": message["content"]
                })
        else:
            anthropic_messages.append({
                "role": "assistant",
                "content": self.conversation[0]["content"]
            })

            for message in messages[2:]:
                anthropic_messages.append({
                    "role": self.role_to_prompt[message["role"]],
                    "content": message["content"]
                })

        return system_prompt, anthropic_messages

    def chat(self):

        if self.api_type == "anthropic":
            t = self.messages_to_prompt(self.conversation)
            completion = self.anthropic.completions.create(
                model=self.model,
                max_tokens_to_sample=400,
                temperature=0.7,
                prompt=t,
            )
            completion = completion.completion
        else:
            system_prompt, messages = self.messages_to_anthropic_format(
                self.conversation)
            response = self.anthropic.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=400,
                temperature=0.7,
                messages=messages
            )
            completion = ""
            for content_block in response.content:
                completion += content_block.text
        time.sleep(0.2)
        return completion

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})
