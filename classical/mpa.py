import torch
from torch import bmm
from tqdm import tqdm
import argparse
from typing import Tuple

from radar import MultiRadar


class MPA:
    def __init__(
        self,
        args: argparse.Namespace,
        multiradar: MultiRadar,
        max_iter: int = 100,
        tol: float = 1e-2,
    ):
        self.args = args
        self.mr = multiradar
        self.max_iter = max_iter
        self.tol = tol

        self.device = "cpu"

        self.N_subbands = multiradar.N_subbands

        self.print_loss = False

    def _mpa_batch_memory_old(
        self, sar_data_LR: torch.Tensor, checkpoint: dict
    ) -> torch.Tensor:
        if checkpoint["first_run"]:
            s_out = torch.zeros_like(sar_data_LR)
            checkpoint["first_run"] = False
        else:
            s_out = torch.from_numpy(checkpoint["checkpoint"]).to(self.device)
        self.s_out = s_out

        for i_gap in range(self.N_subbands - 1):
            s1, s2, s_fb = self._mpa_prep(i_gap, i_gap + 1)

            print(f"Running iterative MPA for subbands {i_gap+1} and {i_gap+2}")
            for i in tqdm(range(checkpoint[f"gap{i_gap}_start"], sar_data_LR.shape[0])):
                s_LR = sar_data_LR[i]
                (
                    s1,
                    s2,
                ) = self._load_signals(s1, s2, s_LR)

                s_out[i, :, s_fb["idx_overall"]] = self._mpa_reconstruct(s1, s2, s_fb)

                checkpoint[f"gap{i_gap}_completed"] = i

        checkpoint["is_completed"] = True

        return s_out

    def _mpa_batch_memory(
        self, sar_data_LR: torch.Tensor, checkpoint: dict, N_batch: int = 1
    ) -> torch.Tensor:
        if checkpoint["first_run"]:
            s_out = torch.zeros_like(sar_data_LR)
            checkpoint["first_run"] = False
        else:
            s_out = torch.from_numpy(checkpoint["checkpoint"]).to(self.device)
        self.s_out = s_out

        for i_gap in range(self.N_subbands - 1):
            s1, s2, s_fb = self._mpa_prep(i_gap, i_gap + 1)

            if N_batch == 1:
                print(f"Running iterative MPA for subbands {i_gap+1} and {i_gap+2}")
                for i in tqdm(
                    range(checkpoint[f"gap{i_gap}_start"], sar_data_LR.shape[0])
                ):
                    s_LR = sar_data_LR[i]
                    (
                        s1,
                        s2,
                    ) = self._load_signals(s1, s2, s_LR)

                    s_out[i, :, s_fb["idx_overall"]] = self._mpa_reconstruct(
                        s1, s2, s_fb
                    )

                    checkpoint[f"gap{i_gap}_completed"] = i
            else:
                print(
                    f"Running batch iterative MPA for subbands {i_gap+1} and {i_gap+2}"
                )
                for i in tqdm(
                    range(
                        checkpoint[f"gap{i_gap}_start"], sar_data_LR.shape[0], N_batch
                    )
                ):
                    s_LR = sar_data_LR[i : i + N_batch]
                    (
                        s1,
                        s2,
                    ) = self._load_signals(s1, s2, s_LR)

                    s_out[i : i + N_batch, :, s_fb["idx_overall"]] = (
                        self._mpa_reconstruct_batch(s1, s2, s_fb, N_batch).view(
                            N_batch, 1, -1
                        )
                    )

                    checkpoint[f"gap{i_gap}_completed"] = i

        checkpoint["is_completed"] = True

        return s_out

    def _mpa_batch_simple(self, sar_data_LR: torch.Tensor) -> torch.Tensor:
        s_out = torch.zeros_like(sar_data_LR)

        for i_subband in range(self.N_subbands - 1):
            s1, s2, s_fb = self._mpa_prep(i_subband, i_subband + 1)

            print(f"Running iterative MPA for subbands {i_subband+1} and {i_subband+2}")
            for i, s_LR in tqdm(enumerate(sar_data_LR)):
                (
                    s1,
                    s2,
                ) = self._load_signals(s1, s2, s_LR)
                s_out[i, :, s_fb["idx_overall"]] = self._mpa_reconstruct(s1, s2, s_fb)

        return s_out

    def _load_signals(
        self, s1: torch.Tensor, s2: torch.Tensor, s_LR: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N_batch = s_LR.shape[0]
        if N_batch == 1:
            s1["signal"] = s_LR[:, s1["idx_overall"]].view(-1, 1)
            s2["signal"] = s_LR[:, s2["idx_overall"]].view(-1, 1)
        else:
            s1["signal"] = s_LR[:, :, s1["idx_overall"]].view(N_batch, -1, 1)
            s2["signal"] = s_LR[:, :, s2["idx_overall"]].view(N_batch, -1, 1)
        return s1, s2

    def _mpa_prep(
        self, ind_radar1: torch.Tensor, ind_radar2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = {
            "M": self.mr.r[ind_radar1].Nk,
            "m": self.mr.idx[ind_radar1] - self.mr.idx[ind_radar1][0],
            "ind_radar": ind_radar1,
            "idx_overall": self.mr.idx[ind_radar1],
        }

        s2 = {
            "M": self.mr.r[ind_radar2].Nk,
            "m": self.mr.idx[ind_radar2] - self.mr.idx[ind_radar1][0],
            "ind_radar": ind_radar2,
            "idx_overall": self.mr.idx[ind_radar2],
        }

        s_fb = {
            "M": s2["m"][-1] - s1["m"][0] + 1,
            "m": torch.arange(s1["m"][0], s2["m"][-1] + 1).to(self.device),
            "idx_between": torch.arange(s1["m"][-1] + 1, s2["m"][0]).to(self.device),
            "idx_overall": torch.arange(
                s1["idx_overall"][0], s2["idx_overall"][-1] + 1
            ).to(self.device),
        }

        return s1, s2, s_fb

    def _mpa_reconstruct(self, s1: dict, s2: dict, s_fb: dict) -> torch.Tensor:
        def loss_func(s1: dict, s2: dict, s_fb: dict) -> torch.Tensor:
            return torch.linalg.norm(
                s1["signal"] - s_fb["signal"][s1["m"]]
            ) + torch.linalg.norm(s2["signal"] - s_fb["signal"][s2["m"]])

        def truncate_svd(
            U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, Nh: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            U = U[:, 0:Nh]
            S = torch.diag(S[0:Nh] + 1j * 0)
            V = Vh[0:Nh, :].H

            return U, S, V

        def compute_P1_P2(si: dict, M: torch.Tensor = None) -> None:
            if M is None:
                M = si["M"]
            Li = M // 3
            Di = torch.zeros(Li + 1, M - Li, dtype=torch.cfloat).to(self.device)

            for indL in range(Li + 1):
                Di[indL, :] = si["signal"][indL : (M - Li + indL)].view(-1)

            si["P1"] = Di[:, 0:Li]
            si["P2"] = Di[:, 1 : (Li + 1)]

        M = min([s1["M"], s2["M"]])

        compute_P1_P2(s1, M)
        compute_P1_P2(s2, M)

        P1 = torch.cat((s1["P1"], s2["P1"]), dim=0)
        P2 = torch.cat((s1["P2"], s2["P2"]), dim=0)

        U1, S1, V1h = torch.linalg.svd(P1)
        U2, S2, V2h = torch.linalg.svd(P2)

        # Estimated number of reflectors (\hat{N}})
        Nh = max([(S1 / S1[0] > 1e-5).sum(), (S2 / S2[0] > 1e-5).sum()])

        # Truncate SVDs
        U1, S1, V1 = truncate_svd(U1, S1, V1h, Nh)
        U2, S2, V2 = truncate_svd(U2, S2, V2h, Nh)

        # Compute Ztn
        Ztn = torch.linalg.eigvals(S1.inverse() @ U1.H @ U2 @ S2 @ V2.H @ V1).view(
            -1, 1
        )

        # Compute ftn
        A = Ztn.view(1, -1) ** torch.arange(s1["M"] + s2["M"]).view(-1, 1).to(
            self.device
        )
        ftn = torch.linalg.pinv(A) @ torch.cat((s1["signal"], s2["signal"]), dim=0)

        # Reconstruct full-band signal
        s_fb["signal"] = (ftn * Ztn ** s_fb["m"].view(1, -1)).sum(dim=0)

        count = 0
        Err_new = loss_func(s1, s2, s_fb)

        # Fill with known values
        s_fb["signal"][s1["m"]] = s1["signal"].view(-1)
        s_fb["signal"][s2["m"]] = s2["signal"].view(-1)
        while True:
            # Find polse for s_fb
            compute_P1_P2(s_fb)
            P1 = s_fb["P1"]
            P2 = s_fb["P2"]

            try:
                U1, S1, V1h = torch.linalg.svd(P1)
                U2, S2, V2h = torch.linalg.svd(P2)
            except:
                break

            # Estimated number of reflectors (\hat{N}}) TODO: should we do this here?
            Nh = max([(S1 / S1[0] > 1e-1).sum(), (S2 / S2[0] > 1e-1).sum()])

            # Truncate SVDs
            U1, S1, V1 = truncate_svd(U1, S1, V1h, Nh)
            U2, S2, V2 = truncate_svd(U2, S2, V2h, Nh)

            # Compute Ztn
            Ztn = torch.linalg.eigvals(S1.inverse() @ U1.H @ U2 @ S2 @ V2.H @ V1).view(
                -1, 1
            )

            try:
                A = Ztn.view(1, -1) ** s_fb["m"].view(-1, 1)
                ftn = torch.linalg.pinv(A) @ s_fb["signal"].view(-1, 1)
            except:
                break

            s_est = {"signal": (ftn * Ztn ** s_fb["m"].view(1, -1)).sum(dim=0)}

            Err_old = Err_new
            Err_new = loss_func(s1, s2, s_est)

            if self.print_loss:
                print(
                    f"Error at iteration {count}: {Err_new:0.4f}, del: {torch.abs(Err_new - Err_old):0.4f}"
                )

            if Err_new > Err_old:
                # print("Error increased!")
                break
            elif torch.abs(Err_new - Err_old) > self.tol:
                s_fb["signal"][s_fb["idx_between"]] = s_est["signal"][
                    s_fb["idx_between"]
                ]

                count += 1
                if count > self.max_iter:
                    # success = False
                    # print("Max iterations reached")
                    break
            else:
                # success = True
                # print("Converged")
                break

        return s_fb["signal"]

    def _mpa_reconstruct_batch(
        self, s1: dict, s2: dict, s_fb: dict, N_batch: int
    ) -> torch.Tensor:
        def loss_func(s1: dict, s2: dict, s_fb: dict):
            return (
                torch.linalg.norm(
                    s1["signal"][batch_idx[~completed_mask]]
                    - s_fb["signal"][:, s1["m"]],
                    ord=2,
                    dim=1,
                )
                + torch.linalg.norm(
                    s2["signal"][batch_idx[~completed_mask]]
                    - s_fb["signal"][:, s2["m"]],
                    ord=2,
                    dim=1,
                )
            ).squeeze()

        def truncate_svd(
            U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, Nh: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            U = U[:, :, 0:Nh]
            S = torch.diag_embed(S[:, 0:Nh] + 1j * 0)
            V = Vh[:, 0:Nh, :].mH

            del Vh

            return U, S, V

        def compute_P1_P2(si: dict, M: torch.Tensor = None, N: int = N_batch) -> None:
            if M is None:
                M = si["M"]
            Li = M // 3
            Di = torch.zeros(
                N, Li + 1, M - Li, dtype=torch.cfloat, requires_grad=False
            ).to(self.device)

            for indL in range(Li + 1):
                Di[:, indL, :] = si["signal"][:, indL : (M - Li + indL)].view(N, M - Li)

            si["P1"] = Di[:, :, 0:Li]
            si["P2"] = Di[:, :, 1 : (Li + 1)]

        # Index tracking
        completed_mask = torch.zeros(N_batch, dtype=torch.bool, requires_grad=False).to(
            self.device
        )
        batch_idx = torch.arange(N_batch, requires_grad=False).to(self.device)
        s_fb_test = s_fb.copy()

        # Start MPA algorithm
        M = min([s1["M"], s2["M"]])

        compute_P1_P2(s1, M)
        compute_P1_P2(s2, M)

        P1 = torch.cat((s1["P1"], s2["P1"]), dim=1)
        P2 = torch.cat((s1["P2"], s2["P2"]), dim=1)

        U1, S1, V1h = torch.linalg.svd(P1)
        U2, S2, V2h = torch.linalg.svd(P2)

        # Estimated number of reflectors (\hat{N}})
        Nh = (
            (
                max(
                    [
                        (S1 / S1[:, 0].view(-1, 1) > 1e-5).sum(),
                        (S2 / S2[:, 0].view(-1, 1) > 1e-5).sum(),
                    ]
                )
                / N_batch
            )
            .floor()
            .int()
        )

        # Truncate SVDs
        U1, S1_inv, V1 = truncate_svd(U1, 1 / S1, V1h, Nh)
        U2, S2, V2 = truncate_svd(U2, S2, V2h, Nh)

        del V1h, V2h

        # Compute Ztn
        Ztn = torch.linalg.eigvals(
            bmm(bmm(bmm(bmm(bmm(S1_inv, U1.mH), U2), S2), V2.mH), V1)
        ).view(N_batch, -1, 1)

        # Compute ftn
        A = Ztn.view(N_batch, 1, -1) ** torch.arange(
            s1["M"] + s2["M"], requires_grad=False
        ).view(1, -1, 1).to(self.device)
        ftn = bmm(torch.linalg.pinv(A), torch.cat((s1["signal"], s2["signal"]), dim=1))

        # Reconstruct full-band signal
        s_fb["signal"] = (
            (ftn * Ztn ** s_fb["m"].view(1, 1, -1)).sum(dim=1).view(N_batch, -1, 1)
        )

        del U1, S1_inv, V1, U2, S2, V2, Ztn, A, ftn, P1, P2, Nh, M

        count = 0
        Err_new = loss_func(s1, s2, s_fb)

        # Fill with known values
        s_fb["signal"][:, s1["m"]] = s1["signal"]
        s_fb["signal"][:, s2["m"]] = s2["signal"]
        while True:
            s_fb_test["signal"] = s_fb["signal"][batch_idx[~completed_mask]]
            N_batch_i = s_fb_test["signal"].shape[0]

            # Find polse for s_fb
            compute_P1_P2(s_fb_test, N=N_batch_i)
            P1 = s_fb_test["P1"]
            P2 = s_fb_test["P2"]

            U1, S1, V1h = torch.linalg.svd(P1)
            U2, S2, V2h = torch.linalg.svd(P2)

            # Estimated number of reflectors (\hat{N}}) TODO: should we do this here?
            Nh = (
                (
                    max(
                        [
                            (S1 / S1[:, 0].view(-1, 1) > 1e-5).sum(),
                            (S2 / S2[:, 0].view(-1, 1) > 1e-5).sum(),
                        ]
                    )
                    / N_batch_i
                )
                .floor()
                .int()
            )

            # Truncate SVDs
            U1, S1_inv, V1 = truncate_svd(U1, 1 / S1, V1h, Nh)
            U2, S2, V2 = truncate_svd(U2, S2, V2h, Nh)

            del V1h, V2h

            # Compute Ztn
            Ztn = torch.linalg.eigvals(
                bmm(bmm(bmm(bmm(bmm(S1_inv, U1.mH), U2), S2), V2.mH), V1)
            ).view(N_batch_i, -1, 1)

            # Compute ftn
            A = Ztn.view(N_batch_i, 1, -1) ** s_fb_test["m"].view(1, -1, 1).to(
                self.device
            )
            ftn = bmm(torch.linalg.pinv(A), s_fb_test["signal"])

            s_est = {
                "signal": (ftn * Ztn ** s_fb_test["m"].view(1, 1, -1))
                .sum(dim=1)
                .view(N_batch_i, -1, 1)
            }

            Err_old = Err_new.clone()
            Err_new[batch_idx[~completed_mask]] = loss_func(s1, s2, s_est)

            old_was_better = Err_old < Err_new
            diff_is_small = torch.abs(Err_old - Err_new) < self.tol
            mask = old_was_better | diff_is_small

            if mask.min():
                # success = True
                # print("Converged")
                break
            else:
                s_fb["signal"][batch_idx[~completed_mask]][:, s_fb["idx_between"]] = (
                    s_est["signal"][:, s_fb["idx_between"]]
                )
                completed_mask = mask

                count += 1
                if count > self.max_iter:
                    # success = False
                    # print("Max iterations reached")
                    break

        return s_fb["signal"]

    def model(self, s_LR: torch.Tensor) -> torch.Tensor:
        if self.args["xyz_str"].lower() == "xz":
            s_LR = torch.fft.ifft(s_LR, dim=2, norm="ortho")

        return self._mpa_batch_simple(s_LR)

    def run_sar_data(
        self, sar_data_LR: torch.Tensor, checkpoint: dict, N_batch: int = 64
    ) -> torch.Tensor:
        Nx, Ny, Nk = sar_data_LR.shape

        sar_data_LR = torch.from_numpy(sar_data_LR).to(self.device).view(Nx * Ny, 1, Nk)

        with torch.no_grad():
            s_out = (
                self._mpa_batch_memory(
                    sar_data_LR, checkpoint=checkpoint, N_batch=N_batch
                )
                .cpu()
                .detach()
                .numpy()
                .reshape(Nx, Ny, Nk)
            )

        return s_out
