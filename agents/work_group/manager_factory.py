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


from .parallel_manager import ParallelManager


class ManagerFactory(object):
    def __init__(self):
        pass

    def create(self, manager_type: str):
        if manager_type == 'parallel':
            return ParallelManager()
        else:
            raise ValueError("Unknown manager type: {}".format(manager_type))
