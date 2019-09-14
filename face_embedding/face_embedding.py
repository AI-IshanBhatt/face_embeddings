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
    return np.average(face_embedding_vectors, axis=0)


def main():
    images_folder = "/images/"  # Master folder for all the images

    if not os.path.exists(images_folder):
        print(f"Image Folder {images_folder} does not exist please volume mount it")
        sys.exit(1)

    print(f"TOTAL FOLDERS FOUND {len(os.listdir(images_folder))}")
    method_name = os.environ.get("METHOD")

    if method_name:
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
    else:
        print(get_average(get_all_cf(images_folder)))


if __name__ == "__main__":
    main()
