import os
import unittest

import numpy as np

from face_embedding import get_average, get_embeddings


class TestFaceEmbedding(unittest.TestCase):

    def test_get_average(self):
        np_arr = np.asarray([[1,2,3], [4,5,6]])
        average_as_list = get_average(np_arr).tolist()

        self.assertEqual([2.5, 3.5, 4.5], average_as_list)

    def test_get_embeddings(self):
        images_folder = "tests/images/GP"
        total_images = len(os.listdir(images_folder))
        path = os.path.abspath(images_folder)

        embeddings = get_embeddings(path)
        self.assertTupleEqual(embeddings.shape, (total_images, 128))

    def test_get_embedding_only_malformed(self):
        images_folder = "tests/images/JR"
        path = os.path.abspath(images_folder)

        embeddings = get_embeddings(path)
        self.assertTupleEqual(embeddings.shape, (0, ))

    def test_get_embedding_one_of_malformed(self):
        images_folder = "tests/images/KM"
        total_images = len(os.listdir(images_folder))
        path = os.path.abspath(images_folder)

        embeddings = get_embeddings(path)
        self.assertTupleEqual(embeddings.shape, (total_images - 1, 128))



