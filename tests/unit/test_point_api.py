import pytest
import torch

from manifold_opt.geometry.inner_product import inner_product
from manifold_opt.geometry.projection import feasible_set_projection
from manifold_opt.geometry.retraction import polar, retraction, retraction_polar, retraction_qr
from manifold_opt.geometry.riemannian_gradient import grad, grad_reg
from manifold_opt.objectives.first_order import wlra_first_order, wlra_reg_first_order
from manifold_opt.objectives.losses import wlra_loss, wlra_loss_reg


def _point(dtype=torch.float64):
    U = torch.eye(3, 2, dtype=dtype)
    x = torch.tensor([2.0, 0.5], dtype=dtype)
    V = torch.eye(4, 2, dtype=dtype)
    return U, x, V


def _data(dtype=torch.float64):
    a = torch.arange(12, dtype=dtype).reshape(3, 4) / 10.0
    w = torch.ones_like(a)
    return a, w


def test_wlra_loss_uses_point_representation():
    point = _point()
    a, w = _data()
    U, x, V = point

    expected = (w * (a - (U * x.unsqueeze(0)) @ V.transpose(0, 1)).pow(2)).sum()

    assert torch.allclose(wlra_loss(point, a, w), expected)


def test_wlra_loss_reg_adds_middle_block_regularizer():
    point = _point()
    a, w = _data()
    lmbda = 0.25

    expected = wlra_loss(point, a, w) + lmbda * point[1].pow(2).sum()

    assert torch.allclose(wlra_loss_reg(point, a, w, lmbda), expected)


def test_grad_returns_tangent_tuple_not_loss_pair():
    point = _point()
    a, w = _data()

    tangent = grad(point, a, w)

    assert isinstance(tangent, tuple)
    assert len(tangent) == 3
    assert tangent[0].shape == point[0].shape
    assert tangent[1].shape == point[1].shape
    assert tangent[2].shape == point[2].shape


def test_grad_reg_changes_only_middle_block_by_regularization_term():
    point = _point()
    a, w = _data()
    lmbda = 0.3

    base = grad(point, a, w)
    regularized = grad_reg(point, a, w, lmbda)

    assert torch.allclose(regularized[0], base[0])
    assert torch.allclose(regularized[1], base[1] + 2.0 * lmbda * point[1])
    assert torch.allclose(regularized[2], base[2])


def test_wlra_first_order_matches_existing_loss_and_gradients():
    point = _point()
    a, w = _data()

    evaluation = wlra_first_order(point, a, w)

    assert torch.allclose(evaluation.objective, wlra_loss(point, a, w))
    assert torch.allclose(evaluation.data_loss, wlra_loss(point, a, w))
    assert all(torch.allclose(actual, expected) for actual, expected in zip(evaluation.riemannian_gradient, grad(point, a, w)))
    assert not hasattr(evaluation, "euclidean_gradient")


def test_wlra_reg_first_order_matches_existing_loss_and_gradients():
    point = _point()
    a, w = _data()
    lmbda = 0.3

    evaluation = wlra_reg_first_order(point, a, w, lmbda)

    assert torch.allclose(evaluation.objective, wlra_loss_reg(point, a, w, lmbda))
    assert torch.allclose(evaluation.data_loss, wlra_loss(point, a, w))
    assert all(torch.allclose(actual, expected) for actual, expected in zip(evaluation.riemannian_gradient, grad_reg(point, a, w, lmbda)))
    assert not hasattr(evaluation, "euclidean_gradient")


def test_feasible_set_projection_preserves_stiefel_blocks_and_projects_middle_block():
    point = _point()
    tangent = (
        torch.full_like(point[0], 0.1),
        torch.tensor([3.0, 4.0], dtype=torch.float64),
        torch.full_like(point[2], -0.2),
    )
    r = 1.0

    projected = feasible_set_projection(point, tangent, r)

    assert torch.equal(projected[0], tangent[0])
    assert torch.equal(projected[2], tangent[2])
    assert torch.linalg.norm(projected[1] + point[1]) <= r + 1e-12


def test_retraction_returns_column_orthonormal_stiefel_blocks():
    point = _point()
    step = (
        torch.full_like(point[0], 0.01),
        torch.tensor([0.1, -0.2], dtype=torch.float64),
        torch.full_like(point[2], -0.02),
    )

    U_next, x_next, V_next = retraction(point, step)

    assert torch.allclose(U_next.transpose(0, 1) @ U_next, torch.eye(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(x_next, point[1] + step[1])
    assert torch.allclose(V_next.transpose(0, 1) @ V_next, torch.eye(2, dtype=torch.float64), atol=1e-12)


def test_polar_retraction_matches_stiefel_polar_formula():
    point = _point()
    xi = (
        torch.tensor([[0.0, 0.01], [-0.01, 0.0], [0.02, -0.03]], dtype=torch.float64),
        torch.tensor([0.1, -0.2], dtype=torch.float64),
        torch.tensor([[0.0, -0.02], [0.02, 0.0], [0.01, -0.01], [-0.03, 0.04]], dtype=torch.float64),
    )
    U, x, V = point
    U_next, x_next, V_next = retraction_polar(point, xi)

    def expected_polar(block, step):
        a = block + step
        gram = torch.eye(block.shape[1], dtype=block.dtype) + step.transpose(0, 1) @ step
        eigvals, eigvecs = torch.linalg.eigh(gram)
        return (a @ eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.transpose(0, 1)

    assert torch.allclose(U_next, expected_polar(U, xi[0]), atol=1e-12)
    assert torch.allclose(x_next, x + xi[1])
    assert torch.allclose(V_next, expected_polar(V, xi[2]), atol=1e-12)
    assert torch.allclose(U_next.transpose(0, 1) @ U_next, torch.eye(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(V_next.transpose(0, 1) @ V_next, torch.eye(2, dtype=torch.float64), atol=1e-12)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend is not available.")
def test_qr_retraction_runs_on_mps_float32():
    point = tuple(block.to(device="mps", dtype=torch.float32) for block in _point(dtype=torch.float32))
    step = (
        torch.full_like(point[0], 0.01),
        torch.tensor([0.1, -0.2], device="mps", dtype=torch.float32),
        torch.full_like(point[2], -0.02),
    )

    U_next, x_next, V_next = retraction_qr(point, step)

    assert U_next.device.type == "mps"
    assert x_next.device.type == "mps"
    assert V_next.device.type == "mps"
    assert U_next.dtype is torch.float32
    assert torch.allclose(
        U_next.transpose(0, 1) @ U_next,
        torch.eye(2, device="mps", dtype=torch.float32),
        atol=1e-4,
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend is not available.")
def test_polar_retraction_preserves_mps_device_with_eigh_fallback():
    point = tuple(block.to(device="mps", dtype=torch.float32) for block in _point(dtype=torch.float32))
    step = (
        torch.tensor([[0.0, 0.01], [-0.01, 0.0], [0.02, -0.03]], device="mps", dtype=torch.float32),
        torch.tensor([0.1, -0.2], device="mps", dtype=torch.float32),
        torch.tensor(
            [[0.0, -0.02], [0.02, 0.0], [0.01, -0.01], [-0.03, 0.04]],
            device="mps",
            dtype=torch.float32,
        ),
    )

    U_next, x_next, V_next = retraction_polar(point, step)

    assert U_next.device.type == "mps"
    assert x_next.device.type == "mps"
    assert V_next.device.type == "mps"
    assert U_next.dtype is torch.float32
    assert torch.allclose(
        V_next.transpose(0, 1) @ V_next,
        torch.eye(2, device="mps", dtype=torch.float32),
        atol=1e-4,
    )


def test_inner_product_is_blockwise_product():
    point = _point()
    u = (
        torch.ones_like(point[0]),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.full_like(point[2], 3.0),
    )
    v = (
        torch.full_like(point[0], 2.0),
        torch.tensor([4.0, 5.0], dtype=torch.float64),
        torch.ones_like(point[2]),
    )

    expected = torch.sum(u[0] * v[0]) + torch.sum(u[1] * v[1]) + torch.sum(u[2] * v[2])

    assert torch.allclose(inner_product(point, u, v), expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda point, a, w: wlra_loss((point[0], point[1]), a, w),
        lambda point, a, w: wlra_loss(point, a[:, :3], w),
        lambda point, a, w: feasible_set_projection(point, grad(point, a, w), -1.0),
        lambda point, a, w: inner_product(point, grad(point, a, w), (grad(point, a, w)[0], grad(point, a, w)[1])),
        lambda point, a, w: polar(point[0], torch.zeros(2, 2, dtype=torch.float64)),
    ],
)
def test_invalid_inputs_fail_loudly(call):
    point = _point()
    a, w = _data()

    with pytest.raises(ValueError):
        call(point, a, w)


def test_nonfinite_inputs_fail_loudly():
    point = _point()
    a, w = _data()
    bad = a.clone()
    bad[0, 0] = torch.nan

    with pytest.raises(ValueError):
        wlra_loss(point, bad, w)
