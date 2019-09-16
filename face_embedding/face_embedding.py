import glob
import os
import sys

from concurrent import futures
from multiprocessing import Pool, cpu_count, Manager

import face_recognition
import numpy as np

from tqdm import tqdm

manager = Manager()
master_list = manager.list()


def get_embeddings(path: str):
    """
    Gets face embedding array for images in given folder
    :param path:
    :return: Array of shape (N, 128) where N is number of images which encodings could be found
    """
    master_encoding = []

    for file_name in glob.glob(f"{path}/*"):
        picture = face_recognition.load_image_file(file_name)
        encodings = face_recognition.face_encodings(picture)
        try:
            master_encoding.append(encodings[0])
        except IndexError:
            pass
    encoded = np.asarray(master_encoding, dtype=np.float32)
    return encoded


def get_embedding_with_manager(path: str):
    """
    Similar to function above but rather than returning it stores output in manager list
    So that it can be used across multiple processes without any fuss.
    :param path:
    :return: None
    """
    master_encoding = []
    for file_name in glob.glob(f"{path}/*"):
        picture = face_recognition.load_image_file(file_name)
        encodings = face_recognition.face_encodings(picture)
        if encodings:
            master_encoding.append(encodings[0])

    if master_encoding:
        master_list.append(np.asarray(master_encoding, dtype=np.float32))


def get_usable_processors():
    return cpu_count() - 1


def get_all_cf(root_path: str, no_of_processors=None) -> np.ndarray:
    """
    It's a method which accumulates all the images' embedding vectors
    It uses concurrent futures with ProcessPools, uses tqdm for some visual flair
    :param root_path: Path where all downloaded images are stored.
    :param no_of_processors: no of processors to be used by multiprocessing
    :return: Face_embedding vectors of all images
    """
    directories = glob.glob(f"{root_path}*")
    all_result = []

    if no_of_processors is None:
        no_of_processors = get_usable_processors()

    with futures.ProcessPoolExecutor(max_workers=no_of_processors) as executor:
        future_to_result = [executor.submit(get_embeddings, directory) for directory in directories]
        tqdm_kwargs = {"total": len(future_to_result),
                       "unit": "it",
                       "unit_scale": True,
                       "leave": True}
        for future in tqdm(futures.as_completed(future_to_result), **tqdm_kwargs):
            result = future.result()
            if result.size:
                all_result.append(future.result())
    return np.vstack(all_result)


def get_all_mp(root_path: str, no_of_processors=None) -> np.ndarray:
    """
    It's a method which accumulates all the images' embedding vectors
    It uses multiprocessing with apply_async , NOT USED IN FINAL SOLUTION
    :param root_path: Path where all downloaded images are stored.
    :param no_of_processors: no of processors to be used by multiprocessing
    :return: Face_embedding vectors of all images
    """

    directories = glob.glob(f"{root_path}*")
    if no_of_processors is None:
        no_of_processors = get_usable_processors()

    pool = Pool(no_of_processors)
    for directory in directories:
        pool.apply_async(get_embedding_with_manager, args=(directory,))

    pool.close()
    pool.join()

    return np.vstack(master_list)


def get_average(face_embedding_vectors: np.ndarray):
    """
    Gets average of face_embedding vectors along row
    :param face_embedding_vectors: all the face embedding vectors of size (N,M)
    :return: Average array of (M,)
    """
    return np.average(face_embedding_vectors, axis=0)


def similarity_index(images_folder: str, celeb1: str, celeb2: str) -> float:
    """
    Gets level of similarity between two celeb's face embeddings, the lower the value the more similarity
    :param images_folder: Root path
    :param celeb1: average face embeddings of celebrity 1
    :param celeb2: average face embeddings of celebrity 2
    :return: float value specifying similarity index
    """
    celeb1_path = os.path.join(images_folder, celeb1)
    celeb2_path = os.path.join(images_folder, celeb2)

    if os.path.exists(celeb1_path) and os.path.exists(celeb2_path):
        all_results = []
        with futures.ProcessPoolExecutor(max_workers=2) as executor:
            future_to_result = [executor.submit(get_embeddings, path) for path in [celeb1_path, celeb2_path]]
            for future in futures.as_completed(future_to_result):
                result = future.result()
                if result.size:
                    all_results.append(future.result())
            if len(all_results) == 2:
                return f"Similarity index between {celeb1} and {celeb2 } is " \
                    f"{np.linalg.norm(all_results[1] - all_results[0]): .3f}"
            else:
                return "One of the celebrity's face embeddings were not found"

    else:
        return "PLEASE PROVIDE VALID CELEBRITY NAMES"


def main():
    images_folder = "/images/"  # Master folder for all the images

    if not os.path.exists(images_folder):
        print(f"Image Folder {images_folder} does not exist please volume mount it")
        sys.exit(1)

    method_name = os.environ.get("METHOD")

    if method_name and method_name.startswith("get_"):  # get embeddings and get average
        celeb_name = os.environ.get("CELEB")
        if not celeb_name:
            raise ValueError("To use get embeddings or average of embeddings you must provide celeb")
        absolute_path = os.path.join(images_folder, celeb_name)

        if not os.path.exists(absolute_path):
            raise ValueError("Given celebrity is not in the dataset")

        else:
            if method_name == "get_embeddings":
                print(get_embeddings(os.path.join(images_folder, celeb_name)))
            elif method_name == "get_average":
                print(get_average(get_embeddings(os.path.join(images_folder, celeb_name))))

    elif method_name == "similarity_index":
        celeb1 = os.environ.get("CELEB1")
        celeb2 = os.environ.get("CELEB2")

        if celeb1 and celeb2:
            print(similarity_index(images_folder, celeb1, celeb2))
        else:
            print("You need to provide two celebs to get similarity index")
    else:
        print(f"TOTAL FOLDERS FOUND {len(os.listdir(images_folder))}")
        print(get_average(get_all_cf(images_folder)))


if __name__ == "__main__":
    main()
