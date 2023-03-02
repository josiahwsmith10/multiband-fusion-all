import torch
import argparse

from model import ComplexModel
from data import MultiRadarData


def SaveModel(args: argparse.Namespace, model: ComplexModel, loss, trainer, PATH: str):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "loss": loss,
        "args": args,
        "train_loss": model.train_loss,
        "val_loss": model.val_loss,
    }

    torch.save(checkpoint, PATH)
    print(f"Saved model to: {PATH}")


class Saver:
    def __init__(self):
        pass

    def Save(
        self, args: argparse.Namespace, model: ComplexModel, loss, trainer, PATH: str
    ):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "loss": loss,
            "args": args,
            "train_loss": model.train_loss,
            "val_loss": model.val_loss,
        }

        torch.save(checkpoint, PATH)
        print(f"Saved model to: {PATH}")

    def Load(
        self,
        args: argparse.Namespace,
        data: MultiRadarData,
        ModelClass,
        LossClass,
        TrainerClass,
        PATH: str,
    ):
        checkpoint = torch.load(PATH)

        # Check that args are aligned with saved model
        args_old = checkpoint["args"]
        assert args.model == args_old.model, "model must be the same as saved data!"
        assert (
            args.xyz_str == args_old.xyz_str
        ), "xyz_str must be the same as saved data!"
        assert args.act == args_old.act, "act must be the same as saved data!"
        assert (
            args.in_channels == args_old.in_channels
        ), "in_channels must be the same as saved data!"
        assert (
            args.out_channels == args_old.out_channels
        ), "out_channels must be the same as saved data!"
        assert (
            args.n_feats == args_old.n_feats
        ), "n_feats must be the same as saved data!"
        assert (
            args.kernel_size == args_old.kernel_size
        ), "kernel_size must be the same as saved data!"
        assert (
            args.res_scale == args_old.res_scale
        ), "res_scale must be the same as saved data!"
        assert (
            args.n_res_blocks == args_old.n_res_blocks
        ), "n_res_blocks must be the same as saved data!"
        assert (
            args.precision == args_old.precision
        ), "precision must be the same as saved data!"

        model = ModelClass(args)

        # Create new loss
        if LossClass is not None:
            loss = LossClass(args)
        else:
            loss = None

        # Create new trainer
        if TrainerClass is not None:
            trainer = TrainerClass(args, data, model, loss)
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            trainer = None

        model.load_state_dict(checkpoint["model_state_dict"])

        if "train_loss" in checkpoint and "val_loss" in checkpoint:
            model.train_loss = checkpoint["train_loss"]
            model.val_loss = checkpoint["val_loss"]

        model.eval()
        print(f"Loaded model from: {PATH}")
        return args, model, loss, trainer

    def LoadModel(self, ModelClass, PATH: str):
        checkpoint = torch.load(PATH)

        args = checkpoint["args"]
        model = ModelClass(args)

        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        print(f"Loaded model from: {PATH}")
        return args, model
