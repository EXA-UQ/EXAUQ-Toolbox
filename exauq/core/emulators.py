from typing import Any
import mogp_emulator as mogp


class MogpEmulator(object):
    def __init__(self, gp: mogp.GaussianProcess):
        self.gp = self._validate_gp(gp)
    
    @staticmethod
    def _validate_gp(gp: Any):
        if not isinstance(gp, mogp.GaussianProcess):
            raise TypeError(
                "Argument 'gp' must be of type GaussianProcess from the "
                "mogp-emulator package"
                )

        return gp
