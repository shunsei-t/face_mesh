from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  face_matrix = detection_result.facial_transformation_matrixes

  annotated_image = np.copy(rgb_image)
  img_width = annotated_image.shape[1]
  img_height = annotated_image.shape[0]


  for i in range(len(face_matrix)):
    mx = face_matrix[i] # 多分人数分入ってる
    print(mx[0, 3], mx[1, 3], mx[2, 3])

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    for i in face_landmarks[-10:]:
      cv2.circle(img=annotated_image, center=[int(i.x*img_width), int(i.y*img_height)], radius=3, color=[0, 0, 255], thickness=-1)

    right_eye = face_landmarks[-5]
    left_eye = face_landmarks[-10]
    cv2.circle(img=annotated_image, center=[int(right_eye.x*img_width), int(right_eye.y*img_height)], radius=3, color=[0, 255, 0], thickness=-1)
    cv2.circle(img=annotated_image, center=[int(left_eye.x*img_width), int(left_eye.y*img_height)], radius=3, color=[0, 255, 0], thickness=-1)

    eye_pose = np.array([(left_eye.x + right_eye.x)/2.0, (left_eye.y + right_eye.y)/2.0, (left_eye.z + right_eye.z)/2.0])

    right_face = face_landmarks[127]
    left_face = face_landmarks[356]
    cv2.circle(img=annotated_image, center=[int(right_face.x*img_width), int(right_face.y*img_height)], radius=3, color=[255, 0, 0], thickness=-1)
    cv2.circle(img=annotated_image, center=[int(left_face.x*img_width), int(left_face.y*img_height)], radius=3, color=[255, 0, 0], thickness=-1)

    face_pose = np.array([(left_face.x + right_face.x)/2.0, (left_face.y + right_face.y)/2.0, (left_face.z + right_face.z)/2.0])

    # print(eye_pose-face_pose)
    # vec = eye_pose-face_pose
    # cv2.circle(img=annotated_image, center=[int(vec[0]*5000 + img_width/2.0), int(vec[1]*5000 + img_height/2.0)], radius=5, color=[255, 255, 255], thickness=-1)


    """ Normarizeできるがあまり意味はなさそう？
    # face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    # face_landmarks_proto.landmark.extend([
    #   landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    # ])

    # keypoints = []
    # for data_point in face_landmarks_proto.landmark:
    #     keypoints.append({
    #                         'X': data_point.x,
    #                         'Y': data_point.y,
    #                         'Z': data_point.z,
    #                         'Visibility': data_point.visibility,
    #                         })

    # for i in keypoints[-10:]:
    #     cv2.circle(img=annotated_image, center=[int(i['X']*img_width), int(i['Y']*img_height)], radius=3, color=[0, 0, 2550], thickness=-1)
    """

  return annotated_image


capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_frame)

    annotated_image = draw_landmarks_on_image(mp_frame.numpy_view(), detection_result)
    # cv2.imshow('face mesh', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('face mesh', annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()