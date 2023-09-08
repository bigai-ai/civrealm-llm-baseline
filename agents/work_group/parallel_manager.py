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


from .manager import Manager
from .base_worker import BaseWorker


class ParallelManager(Manager):
    def __init__(self):
        self.workers = []

    def add_entity(self, entity_type, entity_id):
        pass

    def remove_entity(self, entity_type, entity_id):
        pass

    def act(self, observations, info, is_new_turn):