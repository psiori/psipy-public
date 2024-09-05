# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Tests for :mod:`~psipy.rl.preprocessing.normalization.StackNormalizer`."""


import numpy as np
import pytest

from psipy.rl.preprocessing import StackNormalizer


@pytest.fixture
def positive_data():
    return np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T


@pytest.fixture
def pos_neg_data():
    return np.array([[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]]).T


@pytest.fixture
def tmpzip(tmpdir):
    return str(tmpdir.join("zipfile.zip"))


def test_max(positive_data, pos_neg_data):
    normalizer = StackNormalizer("max").fit(positive_data)
    assert normalizer.center == 0.0
    assert np.array_equal(normalizer.scale, [[10]])
    inverse = normalizer.inverse_transform(normalizer.transform(positive_data))
    assert np.array_equal(positive_data, inverse)
    normalizer.fit(pos_neg_data)
    inverse = normalizer.inverse_transform(normalizer.transform(pos_neg_data))
    assert np.array_equal(pos_neg_data, inverse)


def test_negative_max(positive_data):
    positive_data = np.vstack((positive_data, [[-12.0]]))
    normalizer = StackNormalizer("max").fit(positive_data)
    assert normalizer.center == 0.0
    assert np.array_equal(normalizer.scale, [[12]])
    scaled = normalizer.transform(positive_data)
    assert np.array_equal(
        scaled.round(2).ravel(),
        [0.0, 0.08, 0.17, 0.25, 0.33, 0.42, 0.5, 0.58, 0.67, 0.75, 0.83, -1.0],
    )
    inverse = normalizer.inverse_transform(scaled)
    assert np.array_equal(positive_data, inverse)


def test_std(positive_data, pos_neg_data):
    normalizer = StackNormalizer("std").fit(positive_data)
    assert normalizer.center == 0.0
    assert np.array_equal(normalizer.scale.round(2), [[3.16]])
    inverse = normalizer.inverse_transform(normalizer.transform(positive_data))
    assert np.array_equal(positive_data, inverse)
    normalizer.fit(pos_neg_data)
    inverse = normalizer.inverse_transform(normalizer.transform(pos_neg_data))
    assert np.array_equal(pos_neg_data, inverse)


def test_minmax(positive_data, pos_neg_data):
    normalizer = StackNormalizer("minmax").fit(positive_data)
    inverse = normalizer.inverse_transform(normalizer.transform(positive_data))
    assert np.array_equal(positive_data, inverse)
    normalizer.fit(pos_neg_data)
    inverse = normalizer.inverse_transform(normalizer.transform(pos_neg_data))
    assert np.array_equal(pos_neg_data, inverse)


def test_meanstd(positive_data, pos_neg_data):
    normalizer = StackNormalizer("meanstd").fit(positive_data)
    assert np.array_equal(normalizer.center, [[5]])
    assert np.array_equal(normalizer.scale.round(2), [[3.16]])
    inverse = normalizer.inverse_transform(normalizer.transform(positive_data))
    assert np.array_equal(positive_data, inverse)
    normalizer.fit(pos_neg_data)
    inverse = normalizer.inverse_transform(normalizer.transform(pos_neg_data))
    assert np.array_equal(pos_neg_data, inverse)


def test_meanmax(positive_data, pos_neg_data):
    normalizer = StackNormalizer("meanmax").fit(positive_data)
    assert np.array_equal(normalizer.center, [[5]])
    assert np.array_equal(normalizer.scale, [[5]])
    inverse = normalizer.inverse_transform(normalizer.transform(positive_data))
    assert np.array_equal(positive_data, inverse)
    normalizer.fit(pos_neg_data)
    inverse = normalizer.inverse_transform(normalizer.transform(pos_neg_data))
    assert np.array_equal(pos_neg_data, inverse)

    normalizer = StackNormalizer("meanmax").fit([[300], [700]])
    assert np.array_equal(normalizer.center, [[500]])
    assert np.array_equal(normalizer.scale, [[200]])
    inverse = normalizer.inverse_transform(normalizer.transform(positive_data))
    print(positive_data.astype(float).tolist())
    assert np.array_equal(positive_data, inverse.round(2))
    inverse = normalizer.inverse_transform(normalizer.transform(pos_neg_data))
    assert np.array_equal(pos_neg_data, inverse.round(2))


def test_identity(positive_data, pos_neg_data):
    normalizer = StackNormalizer("identity").fit(positive_data)
    assert normalizer.center == 0
    assert normalizer.scale == 1
    transformation = normalizer.transform(positive_data)
    inverse = normalizer.inverse_transform(transformation)
    assert transformation.tolist() == inverse.tolist() == positive_data.tolist()

    normalizer = StackNormalizer("identity").fit(pos_neg_data)
    assert normalizer.center == 0
    assert normalizer.scale == 1
    transformation = normalizer.transform(pos_neg_data)
    inverse = normalizer.inverse_transform(transformation)
    assert transformation.tolist() == inverse.tolist() == pos_neg_data.tolist()


def test_alter_axis(positive_data):
    plus_10 = positive_data + 10
    positive_data = np.hstack((positive_data, plus_10))
    normalizer = StackNormalizer("meanstd").fit(positive_data)
    assert np.array_equal(normalizer.center, [[5.0, 15.0]])
    assert np.array_equal(normalizer.scale.round(2), [[3.16, 3.16]])
    # Two legal ways to fit on a single axis, stacked vector or flat list.
    normalizer.fit([[0], [1]], axis=1)
    normalizer.fit([0, 1], axis=1)
    assert np.array_equal(normalizer.center, [[5.0, 0.5]])
    assert np.array_equal(normalizer.scale.round(2), [[3.16, 0.5]])


def test_only_vectors(positive_data):
    normalizer = StackNormalizer("meanstd")
    with pytest.raises(ValueError):
        normalizer.fit(np.array([0, 1]))
    # Raises only an AssertionError in transform as assertions can be removed
    # when running productivly.
    with pytest.raises(AssertionError):
        normalizer.transform(np.array([0, 1]))


def test_str(positive_data):
    normalizer = StackNormalizer("max").fit(positive_data)
    assert str(normalizer) == "StackNormalizer: Method max, center [[0]], scale [[10]]"


def test_nd(pos_neg_data):
    expectation = np.array(
        [[-1.58, -1.26, -0.95, -0.63, -0.32, 0.0, 0.32, 0.63, 0.95, 1.26, 1.58]]
    ).T
    normalizer = StackNormalizer("meanstd").fit(pos_neg_data)
    result = normalizer.transform(pos_neg_data)
    print(expectation.shape, result.shape)
    assert len(pos_neg_data.shape) == 2
    assert np.array_equal(result.round(2), expectation)
    pos_neg_data = pos_neg_data[..., None]
    print(normalizer.center.shape, pos_neg_data.shape)
    result = normalizer.transform(pos_neg_data)
    print(result.shape, result)
    assert len(pos_neg_data.shape) == 3
    assert np.array_equal(result.round(2), expectation[..., None])
    pos_neg_data = pos_neg_data[..., None]
    result = normalizer.transform(pos_neg_data)
    assert len(pos_neg_data.shape) == 4
    assert np.array_equal(result.round(2), expectation[..., None, None])


def test_rgb():
    rgb = np.random.random((2, 5, 5, 3))  # BATCH, ROW, COL, RGB
    normalizer = StackNormalizer("meanstd").fit(rgb)
    rgb2 = np.random.random((2, 5, 5, 3, 2))  # BATCH, ROW, COL, RGB, LOOKBACK
    normalizer.transform(rgb2)


def test_saveload(tmpzip, pos_neg_data):
    normalizer = StackNormalizer("max").fit(pos_neg_data)
    normalizer.save(tmpzip)
    normalizer2 = StackNormalizer.load(tmpzip)
    assert np.array_equal(
        normalizer.transform(pos_neg_data), normalizer2.transform(pos_neg_data)
    )
