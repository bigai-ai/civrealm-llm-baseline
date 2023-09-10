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


from .parallel_auto_gpt_agent import ParallelAutoGPTAgent
from .workers import HierarchicalGPTWorker


class HierarchicalGPTAgent(ParallelAutoGPTAgent):
    def __init__(self):
        super().__init__()

    def initialize_workers(self):
        self.strategy_maker = HierarchicalGPTWorker()
        self.workers = {}
    
    def add_entity(self, entity_type, entity_id):
        self.workers[(entity_type, entity_id)] = HierarchicalGPTWorker()

    def get_obs_input_prompt(self, ctrl_type, actor_name, actor_dict, available_actions):
        zoom_in_obs = actor_dict['observations']['minimap']
        zoom_out_obs = actor_dict['observations']['upper_map']

        return f'The {ctrl_type} is {actor_name}.\nThe zoomed-out observation is {zoom_out_obs}.\nThe zoomed-in observation is {zoom_in_obs}.\nThe available actions are {available_actions}. You should choose one of these actions according to the above observations.'


    def make_decisions(self):
        super().make_decisions()