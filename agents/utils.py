def compose_print(x):
    return f"\033[{x}m"


PRINT_RESUME = compose_print(0)
PRINT_ACTION = compose_print(31) + compose_print(42)
PRINT_STEP = compose_print(1) + compose_print(46)
PRINT_CURRENT = compose_print(4) + compose_print(45)


def print_action(*args):
    print(PRINT_ACTION, *args, PRINT_RESUME)


def print_step(*args):
    print(PRINT_STEP, *args, PRINT_RESUME)


def print_current(*args):
    print(PRINT_CURRENT, *args, PRINT_RESUME)
