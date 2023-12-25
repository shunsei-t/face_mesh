# dependencies
```
opencv-python
mediapipe
```

face_landmarkの後ろから10点が瞳孔

```
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
```
https://developers.google.com/mediapipe/solutions/vision/face_landmarker

位置は公開されていない
https://qiita.com/nemutas/items/6321aeca27492baeeb92

detection_result
https://developers.google.com/mediapipe/api/solutions/java/com/google/mediapipe/tasks/vision/facelandmarker/FaceLandmarkerResult

scaleについて
https://developers.google.com/mediapipe/api/solutions/java/com/google/mediapipe/tasks/components/containers/NormalizedLandmark