import torch
import torch.nn as nn

import cvtorch.nn as cvnn
from cvtorch.nn import SplitL1, SplitMSE, PerpLossSSIM


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print("Preparing complex loss function...")

        self.device = args.device
        self.batch_size = args.batch_size

        self.loss = []
        self.loss_module = nn.ModuleList()
        one_benchmark_loss = False
        for loss in args.loss.split("+"):
            weight, loss_type = loss.split("*")
            if loss_type[0] == "i":
                # Loss is applied to intermediate outputs
                intermediate_loss = True
                benchmark_loss = False
                loss_type = loss_type[1:]
            elif loss_type[0] == "b":
                # Loss is benchmark loss, applied to final output
                assert not one_benchmark_loss, "Only one benchmark loss allowed"
                intermediate_loss = False
                benchmark_loss = True
                loss_type = loss_type[1:]
            else:
                intermediate_loss = False
                benchmark_loss = False

            if loss_type == "SMSE":
                loss_function = cvnn.SplitMSE()
            elif loss_type == "SL1":
                loss_function = cvnn.SplitL1()
            elif loss_type == "PerpSSIM":
                loss_function = cvnn.PerpLossSSIM()

            if intermediate_loss:
                loss_type = "i" + loss_type
            if benchmark_loss:
                loss_type = "b" + loss_type

            self.loss.append(
                {
                    "type": loss_type,
                    "weight": float(weight),
                    "value": 0,
                    "function": loss_function,
                }
            )

        if len(self.loss) > 1:
            self.loss.append({"type": "Total", "weight": 0, "function": None})

        for l in self.loss:
            if l["function"] is not None:
                print("\t{:.3f} * {}".format(l["weight"], l["type"]))
                self.loss_module.append(l["function"])

        self.log = torch.Tensor()

        self.loss_module.to(self.device)
        if args.precision == "half":
            self.loss_module.half()

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, "scheduler"):
                l.scheduler.step()

    def get_loss_module(self):
        return self.loss_module

    def start_log(self):
        for l in self.loss:
            l["value"] = 0

    def end_log(self, n_batches: int):
        losses = {}
        for l in self.loss:
            # Intermediate loss
            if l["type"][0] == "i":
                l["value"] /= n_batches
                losses["intermediate_" + l["type"][1:]] = l["value"]

            # Benchmark loss
            elif l["type"][0] == "b":
                l["value"] /= n_batches
                benchmark_loss = 1000 * l["value"]
                losses[l["type"][1:]] = l["value"]

            # Loss at output and total loss (if used)
            else:
                l["value"] /= n_batches
                losses[l["type"]] = l["value"]

        return benchmark_loss, losses

    def forward(self, sr, hr, intermediate=None):
        # sr - super resolution (prediction)
        # hr - high resolution (label)
        losses = []
        for l in self.loss:
            # Intermediate loss
            if (
                l["function"] is not None
                and intermediate is not None
                and l["type"][0] == "i"
            ):
                loss = sum([l["function"](inter, hr) for inter in intermediate])
                effective_loss = l["weight"] * loss
                losses.append(effective_loss)
                l["value"] += effective_loss.item()

            # Benchmark loss
            elif l["function"] is not None and l["type"][0] == "b":
                loss = l["function"](sr, hr)
                effective_loss = l["weight"] * loss
                losses.append(effective_loss)
                l["value"] += effective_loss.item()

            # Loss at output
            elif l["function"] is not None and l["type"][0] != "i":
                loss = l["function"](sr, hr)
                effective_loss = l["weight"] * loss
                losses.append(effective_loss)
                l["value"] += effective_loss.item()

        loss_sum = sum(losses)

        # Store total loss
        if len(self.loss) > 1:
            self.loss[-1]["value"] += loss_sum.item()

        return loss_sum
