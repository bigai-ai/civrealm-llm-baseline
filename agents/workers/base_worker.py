# Copyright (C) 2023  The Freeciv-gym project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.


import time
import random
from abc import ABC, abstractmethod
from typing import Callable

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import Pinecone

from freeciv_gym.freeciv.utils.freeciv_logging import fc_logger
from ..civ_autogpt.utils import num_tokens_from_messages, extract_json, TOKEN_LIMIT_TABLE


class BaseWorker(ABC):
    def __init__(self, model: str):
        self.model = model
        self.dialogue = []
        self.taken_actions_list = []
        self.message = ''

        self.chain: BaseCombineDocumentsChain = None
        self.memory: ConversationSummaryBufferMemory = None
        self.index: Pinecone = None

        self.init_prompts()
        self.init_llm()
        self.init_index()

        self.command_handlers = {}
        self.register_all_commands()

    @abstractmethod
    def init_prompts(self):
        pass

    @abstractmethod
    def init_llm(self):
        """ self.chain and self.memory should be initialized here
        """
        pass

    @abstractmethod
    def init_index(self):
        # self.index should be initialized here
        pass

    @abstractmethod
    def register_all_commands(self):
        pass

    def register_command(self, command: str, handler: Callable):
        self.command_handlers[command] = handler

    @ abstractmethod
    def generate_command(self, prompt: str):
        """ This function prompts the LLM to generate a command.
        The command could be an action for the agent to execute, or a question for the LLM to continue planning.
        """
        pass

    @abstractmethod
    def process_command(self, response: dict, input_prompt: str, avail_action_list: list):
        """ This function processes the command generated by the LLM.

        The function should return the name of the action to be executed, and the prompt_addition to be added to the input_prompt if exception occurs during the process.

        It should also handle the exceptions that may occur during the process. 

        Return
        ------
        exec_action_name: str
            The name of the action to be executed. If the command cannot be processed, return None.
        prompt_addition: str
            The prompt_addition to be added to the input_prompt if exception occurs during the process.
        """
        pass

    def taken_actions_list_needs_update(self, check_content, check_num=3, top_k_characters=None):
        # Check if there are too many recent repeated actions of check_content in the taken_actions_list.
        if len(self.taken_actions_list) < check_num:
            return False
        
        for action in self.taken_actions_list[-check_num:]:
            if action[:top_k_characters] != check_content:
                return False
        
        fc_logger.debug(f'Too many recent {check_content} actions! Please reconsider action.')
        self.taken_actions_list = []
        return True

    def choose_action(self, input_prompt, avail_action_list, interact_timeout=1200):
        exec_action_name = None
        prompt_addition = ''
        start_time = time.time()
        while exec_action_name is None:
            if time.time() - start_time < interact_timeout:
                try:
                    response = self.generate_command(input_prompt + prompt_addition)
                    exec_action_name, prompt_addition = self.process_command(response, input_prompt, avail_action_list)
                except Exception as e:
                    fc_logger.error(f'Error when choosing action: {str(e)}')
                    fc_logger.error(f'input_prompt: {input_prompt}')
                    fc_logger.error(f'dialogue: {str(self.dialogue)}')
                    fc_logger.error('Retying...')
                    continue
            else:
                exec_action_name = random.choice(avail_action_list)
                fc_logger.debug('Timeout, randomly choose:', exec_action_name)
                print('Timeout, randomly choose:', exec_action_name)
                break
        return exec_action_name
    

    # ==============================================================
    # ====================== Index Maintanence =====================
    # ==============================================================
    def get_similiar_docs(self, query, k=2, score=False):
        if score:
            similar_docs = self.index.similarity_search_with_score(query, k=k)
        else:
            similar_docs = self.index.similarity_search(query, k=k)
        return similar_docs

    def get_answer_from_index(self, query):
        similar_docs = self.get_similiar_docs(query)
        while True:
            try:
                fc_logger.debug(f'Querying with similar_docs: {similar_docs}')
                fc_logger.debug(f'Querying with query: {query}')
                answer = self.chain.run(input_documents=similar_docs,
                                        question=query)
                fc_logger.debug(f'Answer: {answer}')
                break
            except Exception as e:
                fc_logger.error('Error in get_answer_from_index')
                fc_logger.error(repr(e))
                print(e)
        return answer


    # ==============================================================
    # ==================== Dialogue Maintanence ====================
    # ==============================================================
    def load_saved_dialogue(self, load_path):
        with open(load_path, "r") as f:
            self.dialogue = eval(f.read())

    def save_dialogue_to_file(self, save_path):
        with open(save_path, "w", encoding='utf-8') as f:
            for message in self.dialogue:
                f.write(str(message) + '\n')
            
    def add_user_message_to_dialogue(self, message):
        self.dialogue.append({'role': 'user', 'content': message})

    def add_assistent_message_to_dialogue(self, message):
        self.dialogue.append({'role': 'assistant', 'content': message})

    def remove_temp_messages_from_dialogue(self, keep_num=3):
        # By default, the first 3 messsages are not temporary.
        # 1 and 2 are initial task descriptions. 3 is a summary of previous messages. 
        while len(self.dialogue) > keep_num:
            self.dialogue.pop(-1)

    def restrict_dialogue(self):
        limit = TOKEN_LIMIT_TABLE[self.model]
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        # TODO: validate that the messages removed are obs and actions
        while num_tokens_from_messages(self.dialogue, self.model) >= limit:
            temp_message = None
            if self.dialogue[-1]['role'] == 'user':
                temp_message = self.dialogue[-1]
            self.remove_temp_messages_from_dialogue(keep_num=2)

            self.add_user_message_to_dialogue(
                'The former chat history can be summarized as: \n' + self.memory.load_memory_variables({})['history'])

            if temp_message is not None:
                self.dialogue.append(temp_message)
    