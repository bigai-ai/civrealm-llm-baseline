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

import os
import openai
import random
import json
import warnings

from freeciv_gym.freeciv.utils.freeciv_logging import fc_logger
from ..civ_autogpt.utils import num_tokens_from_messages, extract_json, TOKEN_LIMIT_TABLE
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI, AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from agents.prompt_handlers.base_prompt_handler import BasePromptHandler

from .base_worker import BaseWorker


class AzureGPTWorker(BaseWorker):
    """
    This agent uses GPT-3 to generate actions.
    """

    def __init__(self, model: str = 'gpt-35-turbo'):
        assert os.environ['OPENAI_API_TYPE'] == 'azure'
        super().__init__(model)

    def init_prompts(self):
        self.prompt_handler = BasePromptHandler()
        self.state_prompt = self._load_state_prompt()
        self.task_prompt = self._load_task_prompt()

    def init_llm(self):
        openai.api_type = os.environ["OPENAI_API_TYPE"]
        openai.api_version = os.environ["OPENAI_API_VERSION"]
        openai.api_base = os.environ["OPENAI_API_BASE"]
        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.deployment_name = os.environ['DEPLOYMENT_NAME']
        llm = AzureChatOpenAI(openai_api_base=openai.api_base,
                              openai_api_version=openai.api_version,
                              openai_api_key=openai.api_key,
                              openai_api_type=openai.api_type,
                              deployment_name=self.deployment_name,
                              temperature=0.7)
        self.chain = load_qa_chain(AzureOpenAI(
            deployment_name=self.deployment_name,
            model_name=self.model),
            chain_type="stuff")
        self.memory = ConversationSummaryBufferMemory(llm=llm,
                                                      max_token_limit=500)

    def init_index(self):
        pinecone.init(api_key=os.environ["MY_PINECONE_API_KEY"],
                      environment=os.environ["MY_PINECONE_ENV"])
        self.index = Pinecone.from_existing_index(
            index_name='langchain-demo', embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))

    def _load_state_prompt(self):
        state_prompt = self.prompt_handler.state_prompt()
        self.add_user_message_to_dialogue(state_prompt)
        return state_prompt

    def _load_task_prompt(self):
        task_prompt = self.prompt_handler.task_prompt()
        self.add_user_message_to_dialogue(task_prompt)
        return task_prompt
    
    def register_all_commands(self):
        self.register_command('finalDecision', self.handle_command_final_decision)
        self.register_command('askCurrentGameInformation', self.handle_command_ask_current_game_information)
        self.register_command('manualAndHistorySearch', self.handle_command_manual_and_history_search)
        
    def handle_command_final_decision(self, command_input, obs_input_prompt, current_avail_actions):
        exec_action = command_input['action']
        prompt_addition = ''
        if exec_action not in current_avail_actions:
            print('Not in the available action list, retrying...')
            prompt_addition = self.prompt_handler.insist_avail_action()
        else:
            self.taken_actions_list.append(command_input['action'])
            if self.taken_actions_list_needs_update('goto', 15, 4):
                prompt_addition = self.prompt_handler.insist_various_actions(action="goto")
                self.taken_actions_list = []
            if self.taken_actions_list_needs_update('keep_activity', 15, 0):
                prompt_addition = self.prompt_handler.insist_various_actions(action="keep_activity")
                self.taken_actions_list = []

        if prompt_addition != '':
            self.update_dialogue_and_communicate(obs_input_prompt + prompt_addition, pop_num=2)
            return None
        else:
            print('exec_action:', exec_action)
            return exec_action
            
    def handle_command_ask_current_game_information(self, command_input, obs_input_prompt, current_avail_actions):
        self.taken_actions_list.append('askCurrentGameInformation')
        return None
    
    def handle_command_manual_and_history_search(self, command_input, obs_input_prompt, current_avail_actions):
        if self.taken_actions_list_needs_update('look_up', 3, 0):
            answer = self.prompt_handler.finish_look_for()
            self.add_user_message_to_dialogue(answer)
            self.taken_actions_list = []
            return None
        
        query = command_input['look_up']
        answer = self.get_answer_from_index(query)
        if random.random() > 0.5:
            answer += self.prompt_handler.finish_look_for()
            
        self.add_user_message_to_dialogue(answer)
        self.memory.save_context({'assistant': query}, {'user': answer})
        self.taken_actions_list.append('look_up')

        return None

    def process_command(self, command_json, obs_input_prompt,
                        current_unit_name, current_avail_actions):
        fc_logger.debug(f'Processing command: {command_json}')
        try:
            command_input = command_json['command']['input']
            command_name = command_json['command']['name']
        except Exception as e:
            print(e)
            print('Not in given json format, retrying...')
            self.update_dialogue_and_communicate(obs_input_prompt +
                                 self.prompt_handler.insist_json(),
                                 pop_num=2)
            return None
        
        if command_name in self.command_handlers:
            return self.command_handlers[command_name](command_input, obs_input_prompt, current_avail_actions)
        else:
            fc_logger.debug(f'Unknown command: {command_name}')
            if random.random() < 0.8:
                self.dialogue.pop(-1)
            else:
                self.add_user_message_to_dialogue('You should only use the given commands!')
            return None

    def query_llm(self, stop=None, temperature=0.7, top_p=0.95):
        fc_logger.debug(f'Querying with dialogue: {self.dialogue}')
        assert openai.api_type == 'azure'
        return openai.ChatCompletion.create(
            deployment_id=self.deployment_name,
            model=self.model,
            messages=self.dialogue)

    def parse_response(self, response):
        fc_logger.debug(f'Parsing response: {response}')
        try:
            ans = extract_json(response['choices'][0]['message']['content'])
        except:
            return response["choices"][0]["message"]
        return {'role': 'assistant', 'content': ans}

    def communicate(self, content):
        self.add_user_message_to_dialogue(content)
        while True:
            try:
                self.restrict_dialogue()
                raw_response = self.query_llm()
                self.message = self.parse_response(raw_response)
                self.dialogue.append(self.message)

                try:
                    response = json.loads(self.message["content"])
                    break
                except Exception as e:
                    fc_logger.debug('Error in communicate: ' + str(e))
                    self.add_user_message_to_dialogue('You should only respond in JSON format as described!')
                    fc_logger.debug('Not response json, retrying...')
                    continue
                
            except Exception as e:
                fc_logger.debug('Error in communicate: ' + str(e))
                fc_logger.debug('dialogue: ' + str(self.dialogue))
                fc_logger.debug('content: ' + content)
                fc_logger.debug('Retying...')
                continue
        
        self.memory.save_context({'user': content}, {'assistant': str(response)})
        return response