import unittest
import torch
from PS5_DIFUM_VISU import PS5_DIFUM_VISU as PS5

t1 = torch.randn(10, 7, 400)
t2 = torch.randn(10, 70, 400)
t3 = torch.randn(9, 7, 400)
t4 = torch.randn(9, 7, 401)
t5 = torch.randn(10, 40)


class compute_cosine_similarities(unittest.TestCase):

    def test_normal_usage(self):
        res = PS5.compute_cosine_similarities(t1, 0)
        self.assertEqual(t1.size(0), len(res))
        self.assertAlmostEqual(res[0], 1.0, places=4)

    def test_normal_usage_2_tensors(self):
        res = PS5.compute_cosine_similarities(t1, 0, t2)
        self.assertEqual(t1.size(0), len(res))

    def test_normal_usage_2_tensors_not_same_first_dim(self):
        res = PS5.compute_cosine_similarities(t1, 0, t3)
        self.assertEqual(t3.size(0), len(res))

    def test_raise_ValueError(self):
        with self.assertRaises(ValueError):
            PS5.compute_cosine_similarities(t1, 0, t4)


class TestGetTSNE(unittest.TestCase):

    def test_get_TSNE_2D(self):
        res = PS5.get_TSNE(t5, component=2)
        self.assertEqual(res.shape, (t5.size(0), 2))

    def test_get_TSNE_3D(self):
        res = PS5.get_TSNE(t1, component=2)
        self.assertEqual(res.shape, (t1.size(0), 2))

    def test_get_TSNE_3_component(self):
        res = PS5.get_TSNE(t1, component=3)
        self.assertEqual(res.shape, (t1.size(0), 3))


if __name__ == '__main__':
    unittest.main()
