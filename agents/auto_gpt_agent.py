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



import random
import time
import os
from freeciv_gym.freeciv.utils.freeciv_logging import fc_logger
from freeciv_gym.freeciv.utils.language_agent_utility import MOVE_NAMES, INVERSE_MOVE_NAMES

from .language_agent import LanguageAgent
from .workers import AzureGPTWorker


class AutoGPTAgent(LanguageAgent):
    def __init__(self):
        super().__init__()

    def initialize_workers(self):
        return AzureGPTWorker()

    def add_entity(self, entity_type, entity_id):
        pass

    def remove_entity(self, entity_type, entity_id):
        pass

    def process_observations_and_info(self, observations, info):
        self.observations = observations
        self.info = info

    def make_decisions(self):
        decision = self.act(self.observations, self.info)
        if decision:
            self.chosen_actions.put(decision)

    def interact_with_llm_within_time_limit(self,
                                            input_prompt,
                                            current_ctrl_obj_name,
                                            avail_action_list,
                                            interact_timeout=120):
        exec_action_name = None
        start_time = time.time()
        while exec_action_name is None:
            end_time = time.time()
            if (end_time - start_time) >= interact_timeout:
                exec_action_name = random.choice(avail_action_list)
                self.workers.dialogue.pop(-1)
                self.workers.dialogue.pop(-1)
                print('overtime, randomly choose:', exec_action_name)
                break
            try:
                response = self.workers.communicate(input_prompt,
                                                    parse_choice_tag=False)
                self.workers.memory.save_context(
                    {'user': input_prompt}, {'assistant': str(response)})
                exec_action_name = self.workers.process_command(
                    response, input_prompt, current_ctrl_obj_name,
                    avail_action_list)
            except Exception as e:
                fc_logger.error('Error in interact_with_llm_within_time_limit')
                fc_logger.error(repr(e))
                fc_logger.error('input_prompt: ' + input_prompt)
                raise e
        return exec_action_name

    def act(self, observations, info):
        available_actions = info['available_actions']
        for ctrl_type in available_actions.keys():
            if ctrl_type == 'unit':
                unit_dict = info['llm_info'][ctrl_type]['unit_dict']
                fc_logger.debug(f'unit_dict: {unit_dict}')
                valid_actor_id, valid_actor_name, valid_action_list = self.get_valid_actor_actions(
                    unit_dict, info, ctrl_type)

                if not valid_actor_id:
                    continue

                current_unit_name = valid_actor_name
                current_unit_obs = info['llm_info'][ctrl_type][valid_actor_id]
                fc_logger.debug(f'current unit obs: {current_unit_obs}')

                current_avail_actions_list = [
                    MOVE_NAMES[action_name]
                    if action_name in MOVE_NAMES.keys() else action_name
                    for action_name in valid_action_list
                ]

                obs_input_prompt = f"""The unit is {current_unit_name}, observation is {current_unit_obs}. Your available action list is {current_avail_actions_list}. """
                print('current unit:', current_unit_name, '; unit id:',
                      valid_actor_id)

                exec_action_name = self.interact_with_llm_within_time_limit(
                    obs_input_prompt, current_unit_name,
                    current_avail_actions_list)

                try:
                    exec_action_name = INVERSE_MOVE_NAMES[exec_action_name]
                except:
                    pass
                if exec_action_name:
                    return (ctrl_type, valid_actor_id, exec_action_name)

            elif ctrl_type == 'city':
                city_dict = info['llm_info'][ctrl_type]['city_dict']
                fc_logger.debug(f'city_dict: {city_dict}')
                valid_actor_id, valid_actor_name, valid_action_list = self.get_valid_actor_actions(
                    city_dict, info, ctrl_type)

                if not valid_actor_id:
                    continue

                current_city_name = valid_actor_name
                current_city_obs = info['llm_info'][ctrl_type][valid_actor_id]
                fc_logger.debug(f'current city obs: {current_city_obs}')

                current_avail_actions_list = valid_action_list

                obs_input_prompt = f"""The city is {current_city_name}, observation is {current_city_obs}. Your available action list is {current_avail_actions_list}. """
                print('current city:', current_city_name, '; city id:',
                      valid_actor_id)

                exec_action_name = self.interact_with_llm_within_time_limit(
                    obs_input_prompt, current_city_name,
                    current_avail_actions_list)

                if exec_action_name:
                    return (ctrl_type, valid_actor_id, exec_action_name)

            else:
                continue

        local_time = time.localtime()
        self.workers.save_dialogue_to_file(
            os.path.join(
                os.getcwd(), "agents/civ_autogpt/saved_dialogues/" +
                f"saved_dialogue_for_T{info['turn'] + 1}_at_{local_time.tm_year}_{local_time.tm_mon}_{local_time.tm_mday}.txt"
            ))
        return None

    def get_valid_actor_actions(self, actor_dict, info, ctrl_type):
        if info['turn'] != self.turn:
            self.planned_actor_ids = []
            self.turn = info['turn']

        for actor in actor_dict:
            actor_name = ' '.join(actor.split(' ')[0:-1])
            actor_id = int(actor.split(' ')[1])

            if actor_id in self.planned_actor_ids:
                continue

            avail_actions = []
            for action_name in actor_dict[actor]['avail_actions']:
                if info['available_actions'][ctrl_type][actor_id][action_name]:
                    avail_actions.append(action_name)

            self.planned_actor_ids.append(actor_id)
            return actor_id, actor_name, avail_actions

        return None, None, None
