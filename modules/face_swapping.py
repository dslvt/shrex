from typing import List, Union, Dict, Set, Tuple
import onnxruntime
from PIL import Image
import os
import insightface
import cv2
import numpy as np
import copy


def get_face_analyser(providers,
                      det_size=(320, 320)):
    """
        Gets an analysis of the face via the insight face python library
        using the model we have
    """
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", root="./models/", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_face_swap_model(model_path: str):
    """
        Fetches the model used in performing face swapping.
    """
    return insightface.model_zoo.get_model(model_path)
    

def swap_face(face_swapper,
              source_faces,
              target_faces,
              source_index,
              target_index,
              temp_frame):
    """
    performs an affine transformation to map source 
    face to a face in the target domain.
    """
    target_face = target_faces[target_index]
    source_face = source_faces[source_index]
    
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def get_many_faces(face_analyser,
                   frame: np.ndarray):
    """
    Handles multiple faces in an image.
    get faces in the order from left to right
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def face_swapping_tool(source_img: Union[Image.Image, List],
                       target_img: Image.Image,
                       source_indexes: str,
                       target_indexes: str,
                       model: str):
    """
        This function handles swapping the faces in the source image to those in the target subspace.
        It works by
            * Analysing the faces in the image
            * fetching the list of faces in the model
            * Replacing the faces
            * returns a new image in the chosen style.
    """
    providers = onnxruntime.get_available_providers()

    face_analyser = get_face_analyser(providers)

    model_path = os.path.join('./models/', model)
    face_swapper = get_face_swap_model(model_path)

    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and num_source_images == num_target_faces:
            print("Replacing faces in target image from the left to the right by order")
            for i in range(num_target_faces):
                source_faces = get_many_faces(face_analyser, cv2.cvtColor(
                    np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                source_index = i
                target_index = i

                if source_faces is None:
                    raise Exception("No source faces found!")

                temp_frame = swap_face(
                    face_swapper,
                    source_faces,
                    target_faces,
                    source_index,
                    target_index,
                    temp_frame
                )
        elif num_source_images == 1:
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(
                np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)
            print(f"Source faces: {num_source_faces}")
            print(f"Target faces: {num_target_faces}")

            if source_faces is None:
                raise Exception("No source faces found!")

            if target_indexes == "-1":
                if num_source_faces == 1:
                    print(
                        "Replacing all faces in target image with the same face from the source image")
                    num_iterations = num_target_faces
                elif num_source_faces < num_target_faces:
                    print(
                        "There are less faces in the source image than the target image, replacing as many as we can")
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    print(
                        "There are less faces in the target image than the source image, replacing as many as we can")
                    num_iterations = num_target_faces
                else:
                    print(
                        "Replacing all faces in the target image with the faces from the source image")
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        face_swapper,
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            else:
                print(
                    "Replacing specific face(s) in the target image with specific face(s) from the source image")

                if source_indexes == "-1":
                    source_indexes = ','.join(
                        map(lambda x: str(x), range(num_source_faces)))

                if target_indexes == "-1":
                    target_indexes = ','.join(
                        map(lambda x: str(x), range(num_target_faces)))

                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception(
                        "Number of source indexes is greater than the number of faces in the source image")

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception(
                        "Number of target indexes is greater than the number of faces in the target image")

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces-1:
                            raise ValueError(
                                f"Source index {source_index} is higher than the number of faces in the source image")

                        if target_index > num_target_faces-1:
                            raise ValueError(
                                f"Target index {target_index} is higher than the number of faces in the target image")

                        temp_frame = swap_face(
                            face_swapper,
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame
                        )
        else:
            raise Exception("Unsupported face configuration")
        result_image = temp_frame
    else:
        print("No target faces found!")

    result_image = Image.fromarray(
        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    return result_image
