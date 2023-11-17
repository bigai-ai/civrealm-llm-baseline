# Configuration mainly for the MastabaAgent Solution

INDIVIDUAL_PROMPT_DEFAULT = True

PROMPT_SOLUTIONS_DICT = {
    "vanilla": "civ_prompts",
    "Settlers": "test_prompts_01_settlers",
    # "city": "",
    "Explorer": "test_prompts_02_explorers",
    "Workers": "test_prompts_03_workers",
    "_final": "civ_prompts",
    # all other entities, should be some "other_prompts"
}


class DictDefaultWrapper(dict):
    """
    Wrapper of dict, for the below use.
    """
    def __getitem__(self, key):
        if key in self.keys():
            return super().__getitem__(key)
        if "_final" in self.keys():
            return super().__getitem__("_final")
        else:
            raise KeyError(
                "`PROMPRT_SOLUTIONS_DICT` in config.py does not have key '_final' defined."
            )


PROMPT_SOLUTIONS = DictDefaultWrapper(PROMPT_SOLUTIONS_DICT)
