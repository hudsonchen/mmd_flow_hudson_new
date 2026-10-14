import torch

from loss import (
    _riesz_topk_softmax_loss_impl,
    _softmax_toward,
)


def test_softmax_toward_matches_materialized_formula():
    torch.manual_seed(17)
    source = torch.randn(2, 4, 7)
    target = torch.randn(2, 5, 7)
    target_weight = torch.rand(2, 5) + 0.25
    tau = 0.7

    distance = torch.cdist(source, target)
    probability = torch.softmax(-distance / tau, dim=-1)
    direction = (target[:, None] - source[:, :, None]) / distance[..., None]
    expected = torch.sum(
        probability[..., None] * target_weight[:, None, :, None] * direction,
        dim=2,
    )

    actual, ess = _softmax_toward(
        source, target, target_weight, tau, direction_epsilon=1e-8
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        ess, 1.0 / torch.sum(probability * probability, dim=-1)
    )


def test_self_softmax_excludes_diagonal_and_loss_has_finite_gradient():
    torch.manual_seed(23)
    gen = torch.randn(2, 4, 6, requires_grad=True)
    fixed_pos = torch.randn(2, 5, 6)
    fixed_neg = torch.randn(2, 3, 6)

    toward, ess = _softmax_toward(
        gen.detach(),
        gen.detach(),
        torch.ones(2, 4),
        tau=0.5,
        direction_epsilon=1e-8,
        exclude_self=True,
    )
    assert torch.isfinite(toward).all()
    assert torch.all(ess >= 1.0)
    assert torch.all(ess <= 3.0 + 1e-6)

    loss, info = _riesz_topk_softmax_loss_impl(
        gen,
        fixed_pos,
        fixed_neg,
        weight_pos=torch.full((2, 5), 2.5),
        weight_neg=torch.full((2, 3), 1.5),
        tau=0.5,
    )
    loss.mean().backward()
    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(gen.grad).all()
    assert info["riesz_topk_softmax_tau"].item() == 0.5


def test_single_generated_particle_has_zero_self_repulsion():
    source = torch.randn(3, 1, 5)
    toward, ess = _softmax_toward(
        source,
        source,
        torch.ones(3, 1),
        tau=1.0,
        direction_epsilon=1e-8,
        exclude_self=True,
    )
    torch.testing.assert_close(toward, torch.zeros_like(source))
    torch.testing.assert_close(ess, torch.zeros_like(ess))


def test_multiscale_drifting_temperatures_are_all_applied():
    torch.manual_seed(29)
    initial_gen = torch.randn(2, 4, 6)
    fixed_pos = torch.randn(2, 5, 6)
    fixed_neg = torch.randn(2, 3, 6)
    temperatures = (0.2, 0.05, 0.02)

    multiscale_gen = initial_gen.clone().requires_grad_(True)
    multiscale_loss, multiscale_info = _riesz_topk_softmax_loss_impl(
        multiscale_gen,
        fixed_pos,
        fixed_neg,
        tau_list=temperatures,
    )
    multiscale_loss.mean().backward()

    assert multiscale_info["riesz_topk_softmax_tau_count"].item() == 3.0
    for temperature in temperatures:
        key = f"riesz_topk_softmax_force_rms_tau_{temperature:g}"
        assert key in multiscale_info
        assert torch.isfinite(multiscale_info[key])
    assert torch.isfinite(multiscale_loss).all()
    assert torch.isfinite(multiscale_gen.grad).all()



def test_combined_velocity_normalization_and_gradient():
    # One source and one positive: raw field is [sum(1/tau), 0].
    for floor in (1e-8, 100.0):
        gen = torch.zeros(1, 1, 2, requires_grad=True)
        pos = torch.tensor([[[2.0, 0.0]]])
        loss, info = _riesz_topk_softmax_loss_impl(
            gen, pos, tau_list=(0.2, 0.05, 0.02), rms_epsilon=floor,
        )
        raw_rms = torch.tensor(75.0 / (2.0 ** 0.5))
        velocity = torch.tensor([[[75.0, 0.0]]]) / raw_rms.clamp_min(floor)
        torch.testing.assert_close(info["riesz_topk_softmax_force_rms"], raw_rms)
        torch.testing.assert_close(
            info["riesz_topk_softmax_velocity_rms"], velocity.square().mean().sqrt(),
        )
        torch.testing.assert_close(loss.mean(), velocity.square().mean())
        loss.mean().backward()
        scale_inputs = (info["scale"] / (2.0 ** 0.5)).clamp_min(1e-3)
        torch.testing.assert_close(gen.grad, -velocity / scale_inputs)


def test_per_temperature_rms_normalizes_before_summing():
    # Every nonzero one-particle temperature field points along x. Drifting's
    # per-temperature normalization makes all three contributions equal.
    gen = torch.zeros(1, 1, 2, requires_grad=True)
    pos = torch.tensor([[[2.0, 0.0]]])
    loss, info = _riesz_topk_softmax_loss_impl(
        gen, pos, tau_list=(0.2, 0.05, 0.02),
        normalize_per_temperature=True,
    )
    expected_velocity = torch.tensor([[[3.0 * 2.0 ** 0.5, 0.0]]])
    torch.testing.assert_close(loss, torch.tensor([9.0]))
    torch.testing.assert_close(
        info["riesz_topk_softmax_velocity_rms"], torch.tensor(3.0)
    )
    loss.mean().backward()
    scale_inputs = (info["scale"] / (2.0 ** 0.5)).clamp_min(1e-3)
    torch.testing.assert_close(gen.grad, -expected_velocity / scale_inputs)


def test_zero_field_has_finite_zero_loss_and_gradient():
    gen = torch.zeros(2, 1, 3, requires_grad=True)
    loss, info = _riesz_topk_softmax_loss_impl(gen, torch.zeros(2, 2, 3))
    loss.mean().backward()
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(gen.grad, torch.zeros_like(gen))
    assert info["riesz_topk_softmax_velocity_rms"].item() == 0.0


def test_rms_floor_must_be_positive():
    for floor in (0.0, -1.0):
        try:
            _riesz_topk_softmax_loss_impl(
                torch.zeros(1, 1, 2), torch.ones(1, 1, 2), rms_epsilon=floor,
            )
        except ValueError as error:
            assert "rms_epsilon" in str(error)
        else:
            raise AssertionError("Non-positive RMS floor was accepted")



def test_dynamic_compile_matches_eager_metrics_loss_and_gradient():
    compiled = torch.compile(
        _riesz_topk_softmax_loss_impl,
        backend="eager",
        dynamic=True,
    )
    # Changing shapes and temperatures must not retain stale metric names.
    for count, kwargs in (
        (4, {"tau_list": (0.2, 0.05, 0.02)}),
        (4, {"tau_list": (0.2, 0.05, 0.02), "normalize_per_temperature": True}),
        (5, {"tau_list": (0.3, 0.1, 0.01)}),
        (3, {"tau": 1.0}),
    ):
        gen = torch.randn(2, count, 6, requires_grad=True)
        compiled_gen = gen.detach().clone().requires_grad_(True)
        pos = torch.randn(2, 7, 6)
        neg = torch.randn(2, 2, 6)
        expected_loss, expected_info = _riesz_topk_softmax_loss_impl(
            gen, pos, neg, **kwargs,
        )
        actual_loss, actual_info = compiled(compiled_gen, pos, neg, **kwargs)
        assert actual_info.keys() == expected_info.keys()
        for key in expected_info:
            torch.testing.assert_close(actual_info[key], expected_info[key])
        torch.testing.assert_close(actual_loss, expected_loss)
        expected_loss.mean().backward()
        actual_loss.mean().backward()
        torch.testing.assert_close(compiled_gen.grad, gen.grad)
