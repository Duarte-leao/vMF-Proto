"""Fast CPU checks; no DINO downloads or datasets required."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from src.data.ap10k import AP10KDataset
from src.engine.fitted import Learner, moment_sum, support_vjp, train_fitted
from src.losses.fitting import feature_anchor, peak_loss, prediction_kl
from src.models.vmf_mp.fitting import (
    ExactConcentration,
    effective_bank,
    exact_vmf_terms,
    log_class_evidence,
    selected_weights,
)
from src.models.vmf_mp.four_loss import FourLossVMF


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(8, 8)
        self.register_buffer("frozen", torch.eye(8))

    def frozen_prefix(self, x):
        return x

    def frozen_patches(self, prefix):
        return prefix

    def adapt_prefix(self, prefix, grid):
        return self.adapter(prefix)

    def forward(self, x):
        return self.adapt_prefix(x, (2, 2)), x, x[:, 0]

    def set_requires_grad(self):
        self.adapter.requires_grad_(True)

    def learnable_parameters(self):
        return list(self.adapter.parameters())


class TinyGate(nn.Module):
    def infer_mask(self, tokens, **kwargs):
        gate = torch.ones(tokens.shape[:2], device=tokens.device)
        gate[:, -1] = 0
        return gate.reshape(len(tokens), 2, 2)


def model(variant, **overrides):
    options = dict(
        backbone=TinyBackbone(),
        fg_extractor=TinyGate(),
        n_classes=2,
        n_prototypes=2,
        dim=8,
        variant=variant,
        shared_prototypes=4,
        kappa_init=5.0,
    )
    options.update(overrides)
    return FourLossVMF(**options)


def configuration():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base=None,
    ):
        cfg = compose(
            config_name="default",
            overrides=[
                "train.device=cpu",
                "train.num_workers=0",
                "train.batch_size=2",
                "train.epochs=3",
                "train.fitting.discovery_until=2",
                "train.fitting.fitted_by=3",
                "train.fitting.minimum_support_images=1",
                "train.fitting.refit_steps=3",
                "train.fitting.micro_batch_size=1",
                "train.fitting.support_microbatch=2",
                "eval.batch_size=2",
            ],
        )
    return cfg


@pytest.mark.parametrize("variant", ["class_specific", "shared"])
def test_readout_matches_exponential_formula(variant):
    torch.manual_seed(2)
    shared = variant == "shared"
    raw = (
        torch.randn(3, 4, 5, dtype=torch.float64)
        if shared
        else torch.randn(3, 4, 2, 3, dtype=torch.float64)
    )
    pi = torch.randn(2, 5 if shared else 3, dtype=torch.float64).softmax(-1)
    gate = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    maximum, winners = raw.max(-1)
    weights = (
        pi.T[winners]
        if shared
        else pi[None]
        .expand(3, -1, -1)
        .gather(2, winners.transpose(1, 2))
        .transpose(1, 2)
    )
    if shared:
        maximum = maximum[..., None]
    oracle = ((1 + weights * (maximum.exp() - 1)) * gate[..., None]).sum(1) / gate.sum(
        -1
    )[:, None]
    torch.testing.assert_close(
        log_class_evidence(raw, gate, pi, shared=shared).exp(), oracle
    )


@pytest.mark.parametrize("variant", ["class_specific", "shared"])
@pytest.mark.parametrize("uniform", [True, False])
def test_mixture_weight_initialization_honors_config(variant, uniform):
    torch.manual_seed(13)
    net = model(variant, pi_init_uniform=uniform)
    torch.manual_seed(13)
    repeated = model(variant, pi_init_uniform=uniform)
    for key in ("pi", "bg_pi"):
        weights = net.bank()[key]
        assert (weights > 0).all()
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        torch.testing.assert_close(weights, repeated.bank()[key], rtol=0, atol=0)
        expected = torch.full_like(weights, 1 / weights.shape[-1])
        if uniform:
            torch.testing.assert_close(weights, expected)
        else:
            assert not torch.equal(weights, expected)


@pytest.mark.parametrize("variant", ["class_specific", "shared"])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_fixed_concentrations_during_statistical_fitting(variant, fraction):
    net = model(variant, kappa_learnable=False, kappa_init=1.0)
    free = net.bank()
    moment = (
        F.normalize(torch.randn_like(free["means"]), dim=-1) * 0.5
    ).requires_grad_()
    mass = torch.full_like(free["kappa"], 10.0)
    bank = effective_bank(free, moment, mass, fraction)
    assert net.kappa_raw.requires_grad is False
    assert bank["kappa_learnable"] is False
    torch.testing.assert_close(bank["kappa"], free["kappa"], rtol=0, atol=0)
    torch.testing.assert_close(bank["kappa"], torch.ones_like(bank["kappa"]))
    if fraction:
        gradient = torch.autograd.grad(bank["means"][:, 0].sum(), moment)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    if fraction == 1:
        torch.testing.assert_close(bank["means"], F.normalize(moment, dim=-1))


@pytest.mark.parametrize("dim", [8, 384, 768])
def test_exact_concentration_and_implicit_gradient(dim):
    r = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, requires_grad=True)
    k = ExactConcentration.apply(r, dim, 100000.0)
    np.testing.assert_allclose(
        exact_vmf_terms(dim, k.detach().numpy())[1], r.detach(), atol=2e-10
    )
    assert torch.autograd.gradcheck(
        lambda x: ExactConcentration.apply(x, dim, 100000.0), (r,), atol=2e-3, rtol=1e-3
    )


def test_image_balanced_positive_population_and_exact_support_gradient():
    torch.manual_seed(4)
    z = torch.randn(3, 4, 8, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 0, 1, 1], [0, 2, 2, 2], [1, 1, 2, 2]])
    weights = selected_weights(torch.rand(3, 4), ids, 3)
    per_image = torch.zeros(3, 3, dtype=torch.float64).scatter_add(1, ids, weights)
    torch.testing.assert_close(per_image, (per_image > 0).double())
    assert (weights > 0).all()
    mass = per_image.sum(0)
    moment = moment_sum(z, ids, weights, 3) / mass[:, None]
    objective = moment.sin().square().sum()
    direct = torch.autograd.grad(objective, z, retain_graph=True)[0]
    gradient = torch.autograd.grad(objective, moment)[0].detach()
    replay = torch.autograd.grad(support_vjp(z, ids, weights, mass, gradient), z)[0]
    torch.testing.assert_close(direct, replay, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("variant", ["class_specific", "shared"])
def test_discovery_fitting_four_losses_and_reload(variant):
    torch.manual_seed(42)
    net, cfg = model(variant), configuration()
    learner = Learner(net, cfg)
    prefix = torch.randn(10, 4, 8)
    cache = dict(
        prefix=prefix,
        noise_prefix=torch.stack((prefix + 0.1, prefix - 0.1)),
        labels=torch.tensor([0, 1] * 5),
        foreground=torch.tensor([[1, 1, 1, 0]] * 10).bool(),
        grid=(2, 2),
    )
    before = net.backbone.adapter.weight.detach().clone()
    frozen = net.backbone.frozen.clone()
    discovery = learner.step(cache, torch.tensor([0, 1]), torch.arange(2, 10), 0.0, 2)
    assert discovery["effective_PPD_weight"] == pytest.approx(0.8)
    for fraction in (0.5, 1.0):
        fitted = learner.step(
            cache, torch.tensor([0, 1]), torch.arange(2, 10), fraction, 3
        )
        for key in (
            "prediction_KL",
            "feature_anchor",
            "reference_peak",
            "noisy_retention",
        ):
            assert np.isfinite(fitted[key]) and fitted[key] >= -1e-10
    assert not torch.equal(before, net.backbone.adapter.weight)
    torch.testing.assert_close(frozen, net.backbone.frozen, rtol=0, atol=0)
    assert fitted["effective_PPD_weight"] == 0.0
    net.eval()
    with torch.no_grad():
        output = net(prefix[:2])
        restored = model(variant).eval()
        restored.load_state_dict(net.state_dict(), strict=True)
        torch.testing.assert_close(
            output["class_logits"], restored(prefix[:2])["class_logits"], rtol=0, atol=0
        )
    assert torch.isfinite(output["maps_raw"]).all()


def test_all_four_auxiliaries_have_finite_gradients():
    torch.manual_seed(8)
    features = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    reference = torch.randn_like(features)
    means = F.normalize(torch.randn(3, 8, dtype=torch.float64), dim=-1).requires_grad_()
    k = torch.full((3,), 8.0, dtype=torch.float64, requires_grad=True)
    presence = torch.ones(2, 3, dtype=torch.bool)
    losses = [
        feature_anchor(features, reference, torch.ones(2, 4).bool()),
        prediction_kl(features[:, 0, :2], torch.rand(2, 2).softmax(-1)),
        peak_loss(features, means, k, reference, means.detach(), presence),
        peak_loss(
            features, means, k, features.detach() + 0.1, means.detach(), presence
        ),
    ]
    for loss in losses:
        gradient = torch.autograd.grad(loss, features, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def ap10k_fixture(root):
    (root / "annotations").mkdir()
    (root / "data").mkdir()
    categories = [
        dict(id=1, name="cat", keypoints=["left_eye", "right_eye"]),
        dict(id=2, name="dog", keypoints=["left_eye", "right_eye"]),
    ]
    for split, image_id in [("train", 1), ("val", 2), ("test", 3)]:
        Image.new("RGB", (16, 16)).save(root / "data" / f"{image_id}.jpg")
        document = dict(
            categories=categories,
            images=[
                dict(id=image_id, file_name=f"{image_id}.jpg", width=16, height=16)
            ],
            annotations=[
                dict(
                    id=image_id,
                    image_id=image_id,
                    category_id=1 if split != "test" else 2,
                    keypoints=[4, 4, 2, 12, 4, 1],
                )
            ],
        )
        (root / "annotations" / f"ap10k-{split}-split1.json").write_text(
            json.dumps(document)
        )


def test_ap10k_official_splits_and_training_interface(tmp_path):
    ap10k_fixture(tmp_path)
    train = AP10KDataset(tmp_path, "train", num_classes=2)
    test = AP10KDataset(tmp_path, "test", num_classes=2)
    assert train.classes == test.classes == ["cat", "dog"]
    assert train.image_ids == [1] and test.image_ids == [3]
    assert len(train[0]) == 3  # image, label, row index; no boxes/keypoints
    assert test[0][1] == 1
    path = tmp_path / "annotations" / "ap10k-test-split1.json"
    document = json.loads(path.read_text())
    document["annotations"].append(dict(image_id=3, category_id=1))
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="ambiguous"):
        AP10KDataset(tmp_path, "test")


@pytest.mark.parametrize("variant", ["class_specific", "shared"])
@pytest.mark.parametrize("kappa_learnable", [True, False])
def test_training_checkpoint_and_resume(tmp_path, monkeypatch, variant, kappa_learnable):
    from src.data import dataloader

    class Dataset(torch.utils.data.Dataset):
        classes = ["a", "b"]

        def __len__(self):
            return 6

        def __getitem__(self, index):
            return (
                torch.randn(4, 8, generator=torch.Generator().manual_seed(index)),
                index % 2,
                index,
            )

    monkeypatch.setattr(dataloader, "build_dataset", lambda *args: Dataset())
    cfg = configuration()
    cfg.dataset.num_classes = 2
    cfg.model.num_prototypes = 2
    cfg.model.variant = variant
    cfg.model.kappa_learnable = kappa_learnable
    net = model(variant, kappa_learnable=kappa_learnable)
    initial_kappa = net.kappa_raw.detach().clone()
    path = train_fitted(cfg, tmp_path, model=net)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 3
    assert checkpoint["monitor"]["test_selected"] is True
    restored = model(variant, kappa_learnable=kappa_learnable).eval()
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert checkpoint["reference"] is not None
    if not kappa_learnable:
        assert net.kappa_raw.requires_grad is False
        assert net.kappa_raw.grad is None
        for key in ("training_state_dict", "state_dict"):
            torch.testing.assert_close(
                checkpoint[key]["kappa_raw"], initial_kappa, rtol=0, atol=0
            )
    cfg.run.resume_dir = str(tmp_path)
    assert train_fitted(
        cfg, tmp_path, model=model(variant, kappa_learnable=kappa_learnable)
    ) == path
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("variant", ["class_specific", "shared"])
def test_resume_rejects_changed_discovery_losses(tmp_path, monkeypatch, variant):
    from omegaconf import OmegaConf
    from src.data import dataloader

    class Dataset(torch.utils.data.Dataset):
        classes = ["a", "b"]

        def __len__(self):
            return 6

        def __getitem__(self, index):
            return (
                torch.randn(4, 8, generator=torch.Generator().manual_seed(index)),
                index % 2,
                index,
            )

    monkeypatch.setattr(dataloader, "build_dataset", lambda *args: Dataset())
    cfg = configuration()
    cfg.dataset.num_classes = 2
    cfg.model.num_prototypes = 2
    cfg.model.variant = variant
    train_fitted(cfg, tmp_path, model=model(variant))
    saved_config = (tmp_path / "config.yaml").read_bytes()
    for key in ("l_ppd_coef", "l_ppd_temp", "l_logdet_vmf_coef", "bg_class_weight"):
        changed = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        changed.run.resume_dir = str(tmp_path)
        changed.train.loss[key] = float(changed.train.loss[key]) + 0.123
        with pytest.raises(
            ValueError, match=r"Resume configuration differs at train\.loss"
        ):
            train_fitted(changed, tmp_path, model=model(variant))
        assert (tmp_path / "config.yaml").read_bytes() == saved_config


@pytest.mark.parametrize("variant", ["class_specific", "shared"])
def test_interrupted_fitting_resume_matches_uninterrupted(
    tmp_path, monkeypatch, variant
):
    from src.data import dataloader

    class Dataset(torch.utils.data.Dataset):
        classes = ["a", "b"]

        def __len__(self):
            return 6

        def __getitem__(self, index):
            return (
                torch.randn(4, 8, generator=torch.Generator().manual_seed(index)),
                index % 2,
                index,
            )

    monkeypatch.setattr(dataloader, "build_dataset", lambda *args: Dataset())
    cfg = configuration()
    cfg.train.epochs = 4
    cfg.dataset.num_classes = 2
    cfg.model.num_prototypes = 2
    cfg.model.variant = variant
    original = model(variant).state_dict()
    full = model(variant)
    full.load_state_dict(original)
    full_path = train_fitted(cfg, tmp_path / "full", model=full)
    interrupted = model(variant)
    interrupted.load_state_dict(original)
    step = Learner.step

    def stop(self, cache, query, support, fraction, epoch):
        if epoch == 4:
            raise RuntimeError("simulated interruption")
        return step(self, cache, query, support, fraction, epoch)

    monkeypatch.setattr(Learner, "step", stop)
    with pytest.raises(RuntimeError, match="simulated"):
        train_fitted(cfg, tmp_path / "interrupted", model=interrupted)
    monkeypatch.setattr(Learner, "step", step)
    cfg.run.resume_dir = str(tmp_path / "interrupted")
    resumed_path = train_fitted(cfg, tmp_path / "interrupted", model=model(variant))
    a = torch.load(full_path, map_location="cpu", weights_only=False)
    b = torch.load(resumed_path, map_location="cpu", weights_only=False)
    for key in a["training_state_dict"]:
        torch.testing.assert_close(
            a["training_state_dict"][key], b["training_state_dict"][key], rtol=0, atol=0
        )
    for key in a["state_dict"]:
        torch.testing.assert_close(
            a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0
        )


def test_query_support_overlap_is_rejected():
    learner = Learner(model("class_specific"), configuration())
    with pytest.raises(ValueError, match="excluded"):
        learner.step({}, torch.tensor([0, 1]), torch.tensor([1, 2]), 1.0, 3)


def test_class_specific_warmup_preserves_absent_class():
    from src.engine.fitted import warmup

    net, cfg = model("class_specific"), configuration()
    cache = dict(
        prefix=torch.randn(4, 4, 8),
        labels=torch.zeros(4, dtype=torch.long),
        foreground=torch.tensor([[1, 1, 1, 0]] * 4).bool(),
        grid=(2, 2),
    )
    absent = net.prototype_mu[2:4].detach().clone()
    warmup(net, cache, cfg)
    torch.testing.assert_close(absent, net.prototype_mu[2:4], rtol=0, atol=0)
