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

import os
import time
from civrealm.freeciv.utils.freeciv_logging import fc_logger
from civrealm.freeciv.utils.language_agent_utility import make_action_list_readable, get_action_from_readable_name

from .language_agent import LanguageAgent
from .workers import AzureGPTWorker
from .utils import print_current, print_action


class AutoGPTAgent(LanguageAgent):
    def __init__(self):
        super().__init__()

    def initialize_workers(self):
        self.workers = AzureGPTWorker()
        self.dialogue_dir = os.path.join(
            os.getcwd(), 'agents/civ_autogpt/saved_dialogues/')
        if not os.path.exists(self.dialogue_dir):
            os.makedirs(self.dialogue_dir)

    def add_entity(self, entity_type, entity_id):
        pass

    def remove_entity(self, entity_type, entity_id):
        pass

    def process_observations_and_info(self, observations, info):
        self.observations = observations
        self.info = info

    def make_decisions(self):
        for ctrl_type in self.info['llm_info'].keys():
            for actor_id, actor_dict in self.info['llm_info'][ctrl_type].items(
            ):
                actor_name = actor_dict['name']
                current_unit_obs = actor_dict['observations']['minimap']
                # available_actions = make_action_list_readable(actor_dict['available_actions'])
                available_actions = actor_dict['available_actions']
                obs_input_prompt = f'The {ctrl_type} is {actor_name}, observation is {current_unit_obs}. Your available action list is {available_actions}. '
                print_current(f'Current {ctrl_type}: {actor_name}')
                exec_action_name = self.workers.choose_action(
                    obs_input_prompt, available_actions)
                # exec_action_name = get_action_from_readable_name(exec_action_name)
                print_action('Action chosen:', exec_action_name)
                if exec_action_name:
                    self.chosen_actions.put(
                        (ctrl_type, actor_id, exec_action_name))

                self.workers.save_dialogue_to_file(
                    os.path.join(
                        self.dialogue_dir,
                        f"dialogue_T{self.info['turn'] + 1}_at_{time.strftime('%Y.%m.%d_%H:%M:%S')}.txt"
                    ))
