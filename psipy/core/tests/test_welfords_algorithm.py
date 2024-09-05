import numpy as np

from psipy.core.welfords import WelfordsAlgorithm


class TestWelfordsAlgorithm:
    @staticmethod
    def test_no_samples():
        wa = WelfordsAlgorithm()
        wa.update([])
        assert wa.num_samples == 0
        assert wa.mean == 0
        assert wa.std == 0

    @staticmethod
    def test_one_sample_input():
        test_one_sample = [1]
        wa = WelfordsAlgorithm()
        wa.update(test_one_sample)
        assert wa.num_samples == len(test_one_sample)
        assert wa.mean == 1
        assert wa.std == 0

    @staticmethod
    def test_two_sample_input():
        test_two_samples = [1, 2]
        wa = WelfordsAlgorithm()
        wa.update(test_two_samples)
        assert wa.num_samples == len(test_two_samples)
        assert wa.mean == np.mean(test_two_samples)
        assert wa.std == np.std(test_two_samples)

    @staticmethod
    def test_one_sample_proper_std_shape():
        value = np.zeros((10, 10))[None, ...]
        wa = WelfordsAlgorithm()
        wa.update(value)
        assert wa.std.shape == value.shape[1:]
        np.testing.assert_equal(wa.std, value[0])  # should be all 0

    @staticmethod
    def test_integer_input():
        test_integers = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        wa = WelfordsAlgorithm()
        wa.update(test_integers)
        assert wa.num_samples == len(test_integers)
        assert wa.mean == np.mean(test_integers)
        assert wa.std == np.std(test_integers)

    @staticmethod
    def test_float_input():
        test_floats = [1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5]
        wa = WelfordsAlgorithm()
        wa.update(test_floats)
        assert wa.mean == np.mean(test_floats)
        assert wa.std == np.std(test_floats)

    @staticmethod
    def test_integer_and_float_input():
        test_integers_and_floats = [1, 1, 1, 2.5, 2.5, 2.5, 3, 3, 3]
        wa = WelfordsAlgorithm()
        wa.update(test_integers_and_floats)
        np.testing.assert_allclose(wa.mean, np.mean(test_integers_and_floats))
        np.testing.assert_allclose(wa.std, np.std(test_integers_and_floats))

    @staticmethod
    def test_array_input():
        test_array = np.array(
            [
                np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
                np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
                np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]]),
                np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]]),
            ]
        )
        wa = WelfordsAlgorithm()
        wa.update(test_array)
        assert wa.num_samples == 6
        np.testing.assert_equal(wa.mean, np.mean(test_array, axis=0))
        np.testing.assert_equal(wa.std, np.std(test_array, axis=0))
