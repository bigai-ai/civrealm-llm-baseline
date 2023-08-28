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
import numpy as np
import time
import json
from freeciv_gym.agents.base_agent import BaseAgent
# from freeciv_gym.agents.controller_agent import ControllerAgent
from freeciv_gym.freeciv.utils.freeciv_logging import fc_logger
from freeciv_gym.configs import fc_args
from agents.civ_autogpt import GPTAgent


class LanguageAgent(BaseAgent):
    def __init__(self, LLM_model = 'gpt-3.5-turbo'):
        super().__init__()
        if "debug.agentseed" in fc_args:
            self.set_agent_seed(fc_args["debug.agentseed"])
        self.gpt_agent = GPTAgent(model = LLM_model)

    def interact_with_llm_within_time_limit(self, input_prompt, current_ctrl_obj_name, avail_action_list, interact_timeout = 120):
        exec_action_name = None
        start_time = time.time()
        while exec_action_name is None:
            end_time = time.time()
            if (end_time - start_time) >= interact_timeout:
                exec_action_name = random.choice(avail_action_list)
                self.gpt_agent.dialogue.pop(-1)
                self.gpt_agent.dialogue.pop(-1)
                print('overtime, randomly choose:', exec_action_name)
                break
            response = self.gpt_agent.communicate(input_prompt, parse_choice_tag = False)
            self.gpt_agent.memory.save_context({'user': input_prompt}, {'assistant': str(response)})
            exec_action_name = self.gpt_agent.process_command(response, input_prompt, current_ctrl_obj_name, avail_action_list)
        return exec_action_name


    def act(self, env, observations, info):
        available_actions = info['available_actions']
        for ctrl_type in available_actions.keys():
            if ctrl_type == 'unit':

                unit_dict = env.get_actors_info(observations, ctrl_type, info)
                fc_logger.debug(f'unit_dict: {unit_dict}')
                valid_actor_id, valid_actor_name, valid_action_dict = self.get_next_valid_actor(info, unit_dict, ctrl_type)
                
                if not valid_actor_id:
                    continue
                
                current_unit_name = valid_actor_name
                current_unit_obs = env.get_tiles_info(observations, ctrl_type, valid_actor_id)
                fc_logger.debug(f'unit current obs: {current_unit_obs}')

                current_avail_actions_list = [env.MOVE_NAMES[action_name] if action_name in env.MOVE_NAMES.keys() else action_name for action_name in valid_action_dict.keys()]

                obs_input_prompt = f"""The unit is {current_unit_name}, observation is {current_unit_obs}. Your available action list is {current_avail_actions_list}. """
                print('current unit:', current_unit_name, '; unit id:', valid_actor_id)
                
                exec_action_name = self.interact_with_llm_within_time_limit(obs_input_prompt, current_unit_name, current_avail_actions_list)
                try:
                    exec_action_name = env.INVERSE_MOVE_NAMES[exec_action_name]
                except:
                    pass
                if exec_action_name:
                    return valid_action_dict[exec_action_name]

            elif ctrl_type == 'city':
                city_dict = env.get_actors_info(observations, ctrl_type, info)
                fc_logger.debug(f'city_dict: {city_dict}')

                valid_city_id, current_city_name, valid_city_actions_list = self.get_next_valid_actor(info, city_dict, ctrl_type)
                
                if not valid_city_id:
                    continue

                current_city_obs = env.get_tiles_info(observations, ctrl_type, valid_city_id)

                current_city_avail_actions_list = [action_name for action_name in valid_city_actions_list.keys()]

                obs_input_prompt = f"""The city is {current_city_name}, observation is {current_city_obs}. Your available action list is {current_city_avail_actions_list}. """
                print('current city:', current_city_name, '; city id:', valid_city_id)

                fc_logger.debug(f'city current obs: {current_city_obs}')

                exec_action_name = self.interact_with_llm_within_time_limit(obs_input_prompt, current_city_name, current_city_avail_actions_list)

                if exec_action_name:
                    return valid_city_actions_list[exec_action_name]

            else:
                continue
        return None
    
    def get_next_valid_actor(self, info, unit_dict, desired_ctrl_type=None):
        """
        Return the first actable actor_id and its valid_action_dict that has not been planned in this turn.
        The v1 version does not have a choosing mechanism, so just chooses the first one like former. 
        But we should rewrite it in the next version.
        """
        # TODO: Do we need the turn variable for Agent class?
        if info['turn'] != self.turn:
            self.planned_actor_ids = []
            self.turn = info['turn']

        available_actions = info['available_actions']
        for ctrl_type in available_actions.keys():
            if desired_ctrl_type and desired_ctrl_type != ctrl_type:
                continue

            action_list = available_actions[ctrl_type]
            for actor_id in action_list.get_actors():
                # TODO: We need to write the choosing mechanism in the following version.
                if actor_id in self.planned_actor_ids:
                    # # We have planned an action for this actor in this turn.
                    # continue_flag = 0
                    # for id in unit_dict.keys():
                    #     if actor_id == int(id.split(' ')[-1]):
                    #         # For those not explorer, we only let them move once.
                    #         if id.split(' ')[0] != 'Explorer':
                    #             continue_flag = 1
                    #             break
                    #         if unit_dict[id]['max_move'] <= 0:
                    #             continue_flag = 1
                    #             break
                    # if continue_flag == 1:
                    #     continue
                    continue

                if action_list._can_actor_act(actor_id):
                    fc_logger.debug(f'Trying to operate actor_id {actor_id} by {ctrl_type}_ctrl')
                    valid_action_dict = action_list.get_actions(actor_id, valid_only=True)
                    if not valid_action_dict:
                        continue

                    fc_logger.debug(f'{ctrl_type}_ctrl: Valid actor_id {actor_id} with valid actions found {valid_action_dict}')
                    self.planned_actor_ids.append(actor_id)
                    actor_name = None
                    for id in unit_dict.keys():
                        if actor_id == int(id.split(' ')[-1]):
                            actor_name = id.split(' ')[0]
                            break
                    return actor_id, actor_name, valid_action_dict

        return None, None, None



