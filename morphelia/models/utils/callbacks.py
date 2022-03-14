from typing import Any, Optional, TypeVar

import pytorch_lightning as pl
import torchmetrics
import wandb
from pytorch_lightning.callbacks import Callback

__all__ = ["LogClassAccuracy"]

STEP_OUTPUT = TypeVar("STEP_OUTPUT")


class LogClassAccuracy(Callback):
    def __init__(
        self,
        num_classes,
        label_list,
        train_output_keys=("pred", "target"),
        log_every_n_epochs=50,
    ):
        super(LogClassAccuracy, self).__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.label_list = label_list
        self.pred, self.target = train_output_keys

        self.valid_acc = torchmetrics.F1(num_classes=num_classes, average="none")
        self.train_acc = self.valid_acc.clone()
        self.test_acc = self.valid_acc.clone()

    def __log_data(self, trainer, name, table, log=False):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0 or log:
            trainer.logger.experiment.log(
                {
                    "global_step": trainer.global_step,
                    "epoch": trainer.current_epoch,
                    name: table,
                }
            )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self.train_acc = self.train_acc.to(outputs[self.pred].device)
        self.train_acc(outputs[self.pred], outputs[self.target])

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused=None
    ) -> None:
        scores = self.train_acc.compute()
        table = wandb.Table(
            data=[[self.label_list[i], score] for i, score in enumerate(scores)],
            columns=["label", "value"],
        )

        self.__log_data(
            trainer,
            "train_class_scores",
            wandb.plot.bar(table, "label", "value", "train/class_scores"),
        )

        self.train_acc.reset()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.valid_acc = self.valid_acc.to(outputs[self.pred].device)
        self.valid_acc(outputs[self.pred], outputs[self.target])

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        scores = self.valid_acc.compute()
        table = wandb.Table(
            data=[[self.label_list[i], score] for i, score in enumerate(scores)],
            columns=["label", "value"],
        )

        self.__log_data(
            trainer,
            "valid_class_scores",
            wandb.plot.bar(table, "label", "value", "valid/class_scores"),
        )

        self.valid_acc.reset()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.test_acc = self.test_acc.to(outputs[self.pred].device)
        self.test_acc(outputs[self.pred], outputs[self.target])

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        scores = self.test_acc.compute()
        table = wandb.Table(
            data=[[self.label_list[i], score] for i, score in enumerate(scores)],
            columns=["label", "value"],
        )

        self.__log_data(
            trainer,
            "test_class_scores",
            wandb.plot.bar(table, "label", "value", "test/class_scores"),
            log=True,
        )

        self.test_acc.reset()
