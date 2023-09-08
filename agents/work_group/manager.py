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
from typing import final


class Manager(ABC):
    def __init__(self):
        pass

    def init_new_turn(self):
        pass

    @abstractmethod
    def add_entity(self, entity_type, entity_id):
        pass

    @abstractmethod
    def remove_entity(self, entity_type, entity_id):
        pass

    def distribute_observations(self, processed_observation):
        for worker in self.workers:
            worker.extract_observation(processed_observation)

    def make_decisions(self):
        # TODO: make this a parallel operation
        for worker in self.workers:
            self.chosen_actions.append(worker.choose_action())

    @final
    def act(self, observations, info, is_new_turn):
        if is_new_turn:
            self.chosen_actions = []
            processed_observation = self.process_observation()
            self.distribute_observations()
            self.make_decisions()

        return self.get_next_chosen_action()
