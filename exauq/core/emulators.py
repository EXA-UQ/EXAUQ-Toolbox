from typing import Any
import mogp_emulator as mogp
from exauq.core.modelling import TrainingDatum


class MogpEmulator(object):
    def __init__(self, gp: mogp.GaussianProcess):
        self._gp = self._validate_gp(gp)
        self._training_data = TrainingDatum.list_from_arrays(
            self._gp.inputs, self._gp.targets
            )
    
    @staticmethod
    def _validate_gp(gp: Any) -> mogp.GaussianProcess:
        if not isinstance(gp, mogp.GaussianProcess):
            raise TypeError(
                "Argument 'gp' must be of type GaussianProcess from the "
                "mogp-emulator package"
                )

        return gp

    @property
    def gp(self) -> mogp.GaussianProcess:
        return self._gp

    @property
    def training_data(self) -> list[TrainingDatum]:
        return self._training_data