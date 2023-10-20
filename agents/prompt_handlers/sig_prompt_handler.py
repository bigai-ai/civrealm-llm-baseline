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
"""
Base Prompt Handler @ FC-LLM-Baseline

BIGAI FC-Gym Group
J. Wang
"""

import os
import re
from freeciv_gym.freeciv.utils.freeciv_logging import fc_logger
from .base_prompt_handler import BasePromptHandler

print(os.getcwd())
PROMPT_ROOT_DIR = "./prompt_collections/"
BASE_DIR = "base_prompts/"


class SIGPromptHandler(BasePromptHandler):
    """
    Prompt Handler for SIG

    `{% variable_name %}` for certain variable names
    `{$ prompt_segment_name $}` for prompt segments defined in the same dir.
    `{& if expr() &}` for python-style if and `{& endif &}` strictly required
    `{& for $x in %A &}` if `A` exists, then iterate over A,
        or use [1,..,n] for numbers,
        or ["a",1,22,"bc"] for combination of numbers and strings
        or [%a,%b,%c] %a for variables, grouped
        or [$seg1(param), $seg2(param)] for other segments!
        ALL the above 4 formats could be used together
        for loop should end with `{& endfor &}`
    """
    def _txt_parser(self, raw: str, template_name: str) -> callable:
        """
        Returns prompt generator.

        For efficiency, try not to parse the raw for every prompt generation.
        """
        pass
