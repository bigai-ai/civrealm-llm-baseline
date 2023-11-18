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
"""
Base Prompt Handler @ Civ-LLMs
"""

import os
import re
from civrealm.freeciv.utils.freeciv_logging import fc_logger
from .base_prompt_handler import BasePromptHandler

print(os.getcwd())
PROMPT_ROOT_DIR = "./prompt_collections/"
BASE_DIR = "base_prompts/"


class SIGPromptHandler(BasePromptHandler):
    """
    Prompt Handler for SIG

    `<% variable_name %>` for certain variable names
    `<$ prompt_segment_name $>` for prompt segments defined in the same dir.
    `<& if expr() &>` for python-style if and `<& endif &>` strictly required
    `<& for $x in %A &>` if `A` exists, then iterate over A,
        or use [1,..,n] for numbers,
        or ["a",1,22,"bc"] for combination of numbers and strings
        or [%a,%b,%c] %a for variables, grouped
        or [$seg1(param), $seg2(param)] for other segments!
        ALL the above 4 formats could be used together
        for loop should end with `<& endfor &>`
    """
    def _txt_parser(self, raw: str, template_name: str) -> callable:
        """
        Returns prompt generator.

        For efficiency, try not to parse the raw for every prompt generation.
        """

        full_list = []
        full_list = re.split(R"(<(?P<symbol>[%$&]))[ ]+(.*?)[ ]+(?P=symbol)>",
                             raw)

        def generator(_raise_empty: bool = False, **kwargs) -> str:
            nonlocal self, raw, template_name

            cur_index = 0
            out_list = []

            # Decorator for do-something
            def do_template(operation):
                nonlocal cur_index, out_list

                def inner(_raise_empty, key, kwargs):
                    nonlocal out_list, flist
                    try:
                        operation(_raise_empty, key, kwargs)
                        # I decide not to use `kwargs.get(key, "")` in order to
                        # log the incidents when key is not provided in args.
                    except KeyError as einfo:
                        fc_logger.error(f"Failed to process {key}" +
                                        f" in generating {template_name}.")
                        fc_logger.error(einfo)
                        print(f"Failed to process {key}" +
                              f" in generating {template_name}.")
                        if _raise_empty:
                            raise

                    return inner

            # Decorator to detect is-something
            def is_template(cap):
                def inner(flist):
                    if len(flist) <= 2:
                        return False

                    return cap(flist)

                return inner

            @is_template
            def is_var(flist):
                return flist[0] == "<%" and flist[1] == "%"

            @do_template
            def do_var(_raise_empty, key, kwargs):
                nonlocal out_list, cur_index
                out_list += [str(kwargs[key])]
                cur_index += 3

            @is_template
            def is_prompt(flist):
                return flist[0] == "<$" and flist[1] == "$"

            @is_template
            def is_if(flist):
                if flist[0] != "<&" or flist[1] != "&":
                    return False
                if flist[2] == "if" or flist[2][:3] == "if ":
                    return True
                return False

            @is_template
            def is_endif(flist):
                if flist[0] != "<&" or flist[1] != "&":
                    return False
                if flist[2] == "endif":
                    return True
                return False

            @is_template
            def is_elif(flist):
                if flist[0] != "<&" or flist[1] != "&":
                    return False
                if flist[4] == "elif" or flist[2][:5] == "elif ":
                    return True
                return False

            @is_template
            def is_else(flist):
                if flist[0] != "<&" or flist[1] != "&":
                    return False
                if flist[2] == "if" or flist[2][:3] == "if ":
                    return True
                return False

            @do_template
            def do_if(_raise_empty, key, kwargs, else_stat=2):
                """else_stat: 1 for else, 3 for if and 5 for else."""
                nonlocal out_list, cur_index
                try:
                    if else_stat == 1:
                        pass
                except:
                    pass

            @is_template
            def is_for(flist):
                return flist[0] == "<&" and flist[1] == "&" and flist[
                    2][:4] == "for "

            stack = []
            while cur_index < len(full_list):
                if is_if(full_list[cur_index:]):
                    if do_if(_raise_empty, full_list[cur_index + 2], kwargs):
                        stack += [(1, cur_index)]
                    else:
                        stack += [(0, cur_index)]
                if is_endif(full_list[cur_index:]):
                    stack.pop(-1)
                cur_index += 1

            # def dfs(**kwargs):
            #     nonlocal out_list, flist

            #     while not is_for(flist):

            #         if is_var(flist):
            #             do_var(_raise_empty, flist[2], kwargs)
            #             continue
            #         if is_if(flist):
            #             do_if(_raise_empty, flist[2:], kwargs)
            #         if is_prompt(flist):
            #             pass
            #     # return out_list

            # dfs(**kwargs)
            return "".join(out_list)

        return generator
