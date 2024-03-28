import dill


def save_state_to_file(state, filename='cli_state.dil'):
    with open(filename, 'wb') as f:
        dill.dump(state, f)


def load_state_from_file(filename='cli_state.dil'):
    try:
        with open(filename, 'rb') as f:
            return dill.load(f)
    except FileNotFoundError:
        return None
