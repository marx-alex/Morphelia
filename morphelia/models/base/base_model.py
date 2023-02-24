import torch

import os


class BaseModel:
    """Base Model for Morphelia's convenience models.

    Morphelia model aim to simplify model initialization, training
    and testing.
    """

    def save_model(
        self,
        path: str,
        fname: str = "model",
        overwrite: bool = False,
    ) -> None:
        """Save the current state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.

        Parameters
        ----------
        path : str
            Path to directory
        fname : str
            Name of file to be saved.
        overwrite : bool
            Overwrite existing data or not.
            If `False` and directory already exists at `path`, error will be raised.
        """
        assert self.model is not None, "No model to save."

        if not os.path.exists(path):
            os.mkdir(path)
        # save the model state dict
        file = os.path.join(path, f"{fname}.pt")
        d = 0
        if not overwrite:
            while os.path.isfile(file) or overwrite:
                file = os.path.join(path, f"{fname}_{d:0000d}.pt")
                d += 1

        torch.save(self.model.state_dict(), file)
