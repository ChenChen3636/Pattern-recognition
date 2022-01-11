import cv2
import mediapipe as mp
import time

pTime = 0  # previous time
cTime = 0  # current time


def video():

    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.01)
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)

    while True:
        ret, img = cap.read()
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)
            # print(result.multi_hand_landmarks)
            imgHeight = img.shape[0]
            imgWidth = img.shape[1]
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(
                        img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                    for i, lm in enumerate(handLms.landmark):
                        xPos = int(lm.x * imgWidth)
                        yPos = int(lm.y * imgHeight)

                        cv2.putText(img, str(i), (xPos-25, yPos+5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                        print(i, xPos, yPos)

            # cTime = time.time()
            # fps = 1/(cTime-pTime)
            # pTime = cTime
            # cv2.putText(img, f"FPS : {int(fps)}", (30, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            cv2.imshow('img', img)
        else:
            break

        if cv2.waitKey(1) == ord('q'):
            break


def img_show():
    img = cv2.imread("img.jpg")

    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

    cv2.imshow("beautifulgirl", img)
    cv2.waitKey(0)


if __name__ == "__main__":

    # video()
    img_show()
