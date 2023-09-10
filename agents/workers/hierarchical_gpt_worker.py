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


from .gpt_worker import AzureGPTWorker


class HierarchicalGPTWorker(AzureGPTWorker):
    def _load_intruction_prompt(self):
        intruction_prompt = self.prompt_handler.hierarchical_instruction_prompt()
        self.add_user_message_to_dialogue(intruction_prompt)
    
    def _load_task_prompt(self):
        task_prompt = self.prompt_handler.hierarchical_task_prompt()
        self.add_user_message_to_dialogue(task_prompt)