import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hand モジュールの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# カメラの初期化
capture = cv2.VideoCapture(0)

while True:
    # カメラからフレームを読み込む
    ret, frame = capture.read()

    # フレームをMediaPipeに渡す
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # 手の検出ができた場合
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # 手のランドマークを使って何か処理を行うことができます
            # ここでは手の各関節の座標を表示しています
            for point, landmark in enumerate(landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # フレームを表示
    cv2.imshow("Hand Tracking", frame)

    # 'q' キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放
capture.release()
cv2.destroyAllWindows()
