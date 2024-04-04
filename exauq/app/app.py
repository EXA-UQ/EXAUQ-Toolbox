from collections.abc import Sequence
from numbers import Real


class App:
    def __init__(self):
        pass

    # TODO: Return type just for illustration, not necessarily final API
    def submit(self, inputs: Sequence[Sequence[Real]]) -> dict[str, tuple[float, ...]]:
        return {str(n): tuple(map(float, x)) for n, x in enumerate(inputs)}

    # TODO: Return type just for illustration, not necessarily final API
    def status(self) -> dict[str, int]:
        return {"9999": 1}

    # TODO: Return type just for illustration, not necessarily final API
    def result(self) -> dict[str, int]:
        return {"9999": 1}
