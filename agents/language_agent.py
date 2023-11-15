# Copyright (C) 2023  The CivRealm project
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

"""
The following is a template to create a language agent by inheriting from LanguageAgent.

class MyLanguageAgent(LanguageAgent):
    def __init__(self):
        super().__init__()

    def initialize_workers(self):
        pass

    def add_entity(self, entity_type, entity_id):
        pass

    def remove_entity(self, entity_type, entity_id):
        pass

    def process_observations_and_info(self, observations, info):
        pass

    def make_decisions(self):
        pass

"""


from abc import ABC, abstractmethod
from typing import final
from queue import Queue

from civrealm.agents.base_agent import BaseAgent



class LanguageAgent(BaseAgent):
    def __init__(self, max_deconflict_depth: int = 1):
        super().__init__()
        self.is_new_turn = False
        self.planned_actor_ids = []
        self.turn = None

        self.entities = {'unit': set(), 'city': set()}
        self.workers = None
        self.initialize_workers()
        self.processed_observations = None
        self.processed_info = None
        self.chosen_actions = Queue()
        self.max_deconflict_depth = max_deconflict_depth
        self.current_deconflict_depth = 0
        self.last_taken_actions = {}
        self.conflict_action_list = []

    @abstractmethod
    def initialize_workers(self):
        pass

    @abstractmethod
    def add_entity(self, entity_type, entity_id):
        pass

    @abstractmethod
    def remove_entity(self, entity_type, entity_id):
        pass

    @abstractmethod
    def process_observations_and_info(self, observations, info):
        pass

    @abstractmethod
    def make_decisions(self):
        pass

    def check_is_new_turn(self, info):
        if info['turn'] != self.turn:
            self.is_new_turn = True
            self.planned_actor_ids = []
            self.turn = info['turn']
        else:
            self.is_new_turn = False

    def get_birth_death_entities(self, info):
        birth_entities = {}
        death_entities = {}
        for entity_type in self.entities:
            if entity_type in info['llm_info']:
                new_entities_set = set(info['llm_info'][entity_type])
                birth_entities[entity_type] = set(info['llm_info'][entity_type]) - self.entities[entity_type]
                death_entities[entity_type] = self.entities[entity_type] - set(info['llm_info'][entity_type])
                self.entities[entity_type] = new_entities_set
        return birth_entities, death_entities

    """
    def get_birth_death_entities(self, observations):
        birth_entities = {}
        death_entities = {}
        for entity_type in self.entities:
            new_entities_set = set(observations[entity_type])
            birth_entities[entity_type] = set(
                observations[entity_type]) - self.entities[entity_type]
            death_entities[entity_type] = self.entities[entity_type] - set(
                observations[entity_type])
            self.entities[entity_type] = new_entities_set
        return birth_entities, death_entities
    """

    def handle_new_entities(self, birth_entities):
        for entity_type in birth_entities:
            for entity_id in birth_entities[entity_type]:
                self.add_entity(entity_type, entity_id)

    def handle_dead_entities(self, death_entities):
        for entity_type in death_entities:
            for entity_id in death_entities[entity_type]:
                self.remove_entity(entity_type, entity_id)

    def handle_new_turn(self, observations, info):
        self.process_observations_and_info(observations, info)
        birth_entities, death_entities = self.get_birth_death_entities(info)
        self.handle_new_entities(birth_entities)
        self.handle_dead_entities(death_entities)

        self.chosen_actions = Queue()
        self.make_decisions()

        self.is_new_turn = False

    def handle_conflict_actions(self, action):
        """Handle conflict actions."""

    def regenerate_conflict_actions(self, observations, info):
        """Let LLM rethink on actions which could not be performed."""
        self.process_observations_and_info(observations, info)
        self.conflict_action_list = []

    def is_action_valid(self, info, action):
        ctrl_type, actor_id, action_name = action
        action_dict = info['llm_info'][ctrl_type][actor_id]['available_actions']

        if action_name in action_dict:
            return action_name

        return False

    @final
    def act(self, observations, info):
        self.check_is_new_turn(info)
        if self.is_new_turn:
            self.current_deconflict_depth = 0
            self.handle_new_turn(observations, info)

        while self.current_deconflict_depth < self.max_deconflict_depth:
            if self.chosen_actions.empty():
                self.regenerate_conflict_actions(observations, info)
                self.current_deconflict_depth += 1
            if self.chosen_actions.empty():
                return None
            while not self.chosen_actions.empty():
                action = self.chosen_actions.get()
                if self.is_action_valid(info, action):
                    self.last_taken_actions[tuple(
                        action[:2])] = [action[2], self.turn]
                    return action
                self.handle_conflict_actions(action)

        return None
