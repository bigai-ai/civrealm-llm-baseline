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

import pickle
import os
import time
from .parallel_auto_gpt_agent import ParallelAutoGPTAgent
from .workers import HierarchicalGPTWorker
from agents.redundants.improvement_consts import UNIT_TYPES, IMPR_TYPES

PROD_KINDS = ["improvement", "unit"]
PROD_REF = IMPR_TYPES + UNIT_TYPES


class HierarchicalGPTAgent(ParallelAutoGPTAgent):
    def __init__(self):
        super().__init__()
        self.general_advise = ""

    def initialize_workers(self):
        self.strategy_maker = HierarchicalGPTWorker(role="advisor")
        self.workers = {}

    def add_entity(self, entity_type, entity_id):
        self.workers[(entity_type, entity_id)] = HierarchicalGPTWorker()

    def get_obs_input_prompt(self, ctrl_type, actor_name, actor_dict,
                             available_actions):

        zoom_in_obs = actor_dict['observations']['minimap']
        zoom_out_obs = actor_dict['observations']['upper_map']
        if ctrl_type == "city":

            current_prod = "The city is currently building the "
            kind = self.observations['city'][int(
                actor_name.split()[-1])]['production_kind'] // 3 - 1
            print("CITY", actor_name, kind)
            print(
                f"{PROD_REF[self.observations['city'][int(actor_name.split()[-1])]['production_value']]}"
            )
            current_prod += f"{PROD_KINDS[kind]}"
            current_prod += f"{PROD_REF[self.observations['city'][int(actor_name.split()[-1])]['production_value']]}"
            available_actions += "keep activity"
        else:
            current_prod = ""

        return f'{self.general_advise}\n The {ctrl_type} is {actor_name}.\nThe zoomed-out observation is {zoom_out_obs}.\nThe zoomed-in observation is {zoom_in_obs}.\n{current_prod}\nThe available actions are {available_actions}. You should choose one of these actions according to the above observations.'

    def get_advisor_input_prompt(self, obs, info):
        """
        Generate input prompt for advisor.
        """
        munit_num_self = 0  # millitary units
        wunit_num_self = 0  # working units
        unit_num_enemy = 0
        city_num_self = 0
        city_size_self = 0
        city_num_other = 0
        city_num_enemy = 0
        units = {}
        for key, val in obs['unit'].items():
            if key in info['llm_info']['unit'].keys():
                info_val = info['llm_info']['unit'][key]
                unit_name = info_val["name"].split()[0]
                units[unit_name] = units.get(unit_name, 0) + 1
                if val['type_attack_strength'] == 0:
                    wunit_num_self += 1
                else:
                    munit_num_self += 1

            if obs['dipl'][val['owner']]['diplomatic_state'] == 1:
                unit_num_enemy += 1

        for key, val in obs['city'].items():
            if key in info['llm_info']['city']:
                city_num_self += 1
                city_size_self += val['size']
                continue
            if obs['map']['status'][val['x'], val['y']] <= 1:
                # if a city is not in current view (but in war fog)
                # it is not counted.
                continue
            if obs['dipl'][val['owner']]['diplomatic_state'] == 1:
                city_num_enemy += 1
                continue
            city_num_other += 1

        # print(munit_num_self, wunit_num_self, unit_num_enemy, city_num_self,
        #       city_size_self, city_num_other, city_num_enemy)

        # hand written conditions, change it later.
        if (unit_num_enemy > munit_num_self / 5
                and city_num_enemy < unit_num_enemy):
            war_state = "We are under attack."
        elif (city_num_enemy >= 3 and unit_num_enemy < munit_num_self):
            war_state = "We are attacking other players."
        elif unit_num_enemy == 0 and city_num_enemy < 3:
            war_state = "We are in peace."
        else:
            war_state = "We are roughly safe."

        unit_spec_prompt = (
            f"We have {wunit_num_self+munit_num_self} units: " +
            ", ".join(map(lambda x: f"{x[0]} amount {x[1]}", units.items())))
        return_prompt = " ".join([
            unit_spec_prompt, f"and we can see {unit_num_enemy} enemy units.",
            f"We have {city_num_self} cities of total size {city_size_self}.",
            f"We can see {city_num_enemy} enemy cities, ",
            f"and {city_num_other} other cities.", war_state
        ])
        print(return_prompt)
        return return_prompt

    def generate_general_advise(self):
        """
        Generate general advise for all other workers.
        """

        obs_input_prompt = self.get_advisor_input_prompt(
            self.observations, self.info)

        # obs_input_prompt = self.strategy_maker.prompt_handler.generate(
        #     "advisor_advise")
        print("OBS_INPUT_PROMPT", obs_input_prompt)
        exec_action_name = self.strategy_maker.choose_action(
            obs_input_prompt, ["suggestion"])
        print("GENERAL_ADVISE", exec_action_name)
        self.strategy_maker.save_dialogue_to_file(
            os.path.join(
                self.dialogue_dir,
                f"dialogue_T{self.info['turn'] + 1}_advisor_at_{time.strftime('%Y.%m.%d_%H:%M:%S')}.txt"
            ))
        return exec_action_name

    def make_decisions(self):
        self.general_advise = self.generate_general_advise()
        super().make_decisions()
