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


from abc import ABC, abstractmethod

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
    def query(self):
        pass

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
                break
            except Exception as e:
                print(e)
        return answer

    def load_saved_dialogue(self, load_path):
        with open(load_path, "r") as f:
            self.dialogue = eval(f.read())

    def save_dialogue_to_file(self, save_path):
        with open(save_path, "w", encoding='utf-8') as f:
            for message in self.dialogue:
                f.write(str(message) + '\n')
            
    def add_user_message_to_dialogue(self, message):
        self.dialogue.append({'role': 'user', 'content': message})

    def restrict_dialogue(self):
        limit = TOKEN_LIMIT_TABLE[self.model]
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        # TODO: validate that the messages removed are obs and actions
        while num_tokens_from_messages(self.dialogue, self.model) >= limit:
            temp_message = {}
            user_tag = 0
            if self.dialogue[-1]['role'] == 'user':
                temp_message = self.dialogue[-1]
                user_tag = 1

            while len(self.dialogue) >= 3:
                self.dialogue.pop(-1)

            while True:
                try:
                    self.add_user_message_to_dialogue(
                        'The former chat history can be summarized as: \n' + self.memory.load_memory_variables({})['history'])
                    break
                except Exception as e:
                    print(e)

            if user_tag == 1:
                self.dialogue.append(temp_message)
                user_tag = 0