import pytest
import numpy as np
from sklearn.metrics import mean_squared_error as mse, r2_score

@pytest.fixture()
def load_ys(request):
    file_name = request.param
    ys = np.loadtxt(file_name, delimiter=',')
    return ys

@pytest.fixture()
def load_approx():
    approx = np.loadtxt('datasets/approx.csv', delimiter=',')
    return approx

@pytest.mark.parametrize('load_ys', ['datasets/ys1.csv', 'datasets/ys2.csv', 'datasets/ys3.csv', 'datasets/ys_noisy.csv'], indirect=True)
def test_mse(load_ys, load_approx):
    assert mse(load_ys, load_approx) < 0.5, mse(load_ys, load_approx)

@pytest.mark.parametrize('load_ys', ['datasets/ys1.csv', 'datasets/ys2.csv', 'datasets/ys3.csv', 'datasets/ys_noisy.csv'], indirect=True)
def test_r2(load_ys, load_approx):
    assert r2_score(load_ys, load_approx) > 0.9, r2_score(load_ys, load_approx)
