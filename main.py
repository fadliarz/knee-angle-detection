import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # hide deprecation warning

count = 0
cap= cv2.VideoCapture(1)
RMSE = 0.35 # tolerable error
rc = np.array([[0, 0], [290, 0], [0, 100], [290, 100]])
rc2 = np.zeros((4, 2))
ic = np.zeros((4, 2))
ic2 = np.zeros((4, 2))
L = np.zeros(8)


markers = {
    "first_end": {
        "lower_bound": np.array([55, 83, 62]),
        "upper_bound": np.array([75, 103, 142]),
    },
    "middle": {
        "lower_bound": np.array([-9, 194, 177]),
        "upper_bound": np.array([11, 214, 257]),
    },
    "last_end": {
        "lower_bound": np.array([108, 104, 67]),
        "upper_bound": np.array([128, 124, 147]),
    },
}


def find_angle(first_point, middle_point, last_point):
    x1, y1 = first_point
    x2, y2 = middle_point
    x3, y3 = last_point

    v1x, v1y = (x1 - x2), (y1 - y2)
    v2x, v2y = (x3 - x2), (y3 - y2)

    angle = (
        np.arccos(
            ((v1x * v2x + v1y * v2y))
            / (np.sqrt(v1x**2 + v1y**2) * np.sqrt(v2x**2 + v2y**2))
        )
        * 180
        / np.pi
    )

    return angle


def find_color(mask):
    x_medium = 0
    y_medium = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        x_medium = int((x + x + w) / 2)
        y_medium = int((y + y + h) / 2)
        break

    return x_medium, y_medium


def pick_coordinate(event, x, y, flags, param):
    global count
    global L
    global rc2
    global ic
    global ic2

    if event == cv2.EVENT_LBUTTONDOWN:
        if count < 4:
            ic[count] = [x, y]
        else:
            ic2[count%4] = [x, y]
            xr, yr = reconstruct(L, x, y)
            rc2[count%4] = [xr, yr]

        count += 1
        print(f"count: ", count, (x, y))


def calibrate(rc):
    global L
    global count

    n, _ = rc.shape

    while count < 4:
        _, frame = cap.read()

        cv2.imshow("calibration", frame)
        cv2.setMouseCallback("calibration", pick_coordinate)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cv2.destroyAllWindows()

    A = np.zeros((2 * n, 8))
    b = np.zeros((2 * n, 1))

    for i in range(1, 1 + n):
        b[2 * i - 1 - 1, 0] = ic[i - 1, 0]
        b[2 * i - 1, 0] = ic[i - 1, 1]

        A[2 * i - 1 - 1, 0] = rc[i - 1, 0]
        A[2 * i - 1 - 1, 1] = rc[i - 1, 1]
        A[2 * i - 1 - 1, 2] = 1
        A[2 * i - 1 - 1, 6] = -rc[i - 1, 0] * ic[i - 1, 0]
        A[2 * i - 1 - 1, 7] = -rc[i - 1, 1] * ic[i - 1, 0]
        A[2 * i - 1, 3] = rc[i - 1, 0]
        A[2 * i - 1, 4] = rc[i - 1, 1]
        A[2 * i - 1, 5] = 1
        A[2 * i - 1, 6] = -rc[i - 1, 0] * ic[i - 1, 1]
        A[2 * i - 1, 7] = -rc[i - 1, 1] * ic[i - 1, 1]

    Ap = np.linalg.inv(A)
    L = np.dot(Ap, b)

    while count < 8:
        _, frame = cap.read()

        cv2.imshow("calibration", frame)
        cv2.setMouseCallback("calibration", pick_coordinate)

        key = cv2.waitKey(1)

        if key == 27:
            break
    count = 0

    return L, calculate_rmse(rc, rc2)


def reconstruct(L, x, y):
    A_recon = np.zeros((2, 2))
    b_recon = np.zeros((2, 1))

    A_recon[0, 0] = x * L[6] - L[0]
    A_recon[0, 1] = x * L[7] - L[1]
    A_recon[1, 0] = y * L[6] - L[3]
    A_recon[1, 1] = y * L[7] - L[4]

    b_recon[0, 0] = L[2] - x
    b_recon[1, 0] = L[5] - y

    rc_recon = np.dot(np.linalg.inv(A_recon), b_recon)

    return (rc_recon[0, 0], rc_recon[1,0])


def calculate_rmse(rc, rc2):
    n, _ = ic.shape
    RMSE_X = 0
    RMSE_Y = 0

    for i in range(0, n):
        RMSE_X += (rc[i, 0] - rc2[i, 0]) ** 2
        RMSE_Y += (rc[i, 1] - rc2[i, 1]) ** 2

    RMSE_X = np.sqrt(RMSE_X / (n - 1))
    RMSE_Y = np.sqrt(RMSE_Y / (n - 1))

    return (RMSE_X, RMSE_Y)


def main():
    while True:
        L, (RMSE_X, RMSE_Y) = calibrate(rc)
        print(f"RMSE_X, RMSE_Y = ", (RMSE_X, RMSE_Y))
        print(f"DLT = ", L)

        if RMSE_X <= RMSE and RMSE_Y <= RMSE:
            break

    while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # first end
        first_mask = cv2.inRange(
            hsv_frame,
            markers["first_end"]["lower_bound"],
            markers["first_end"]["upper_bound"],
        )
        xf_medium, yf_medium = find_color(first_mask)
        # middle
        middle_mask = cv2.inRange(
            hsv_frame,
            markers["middle"]["lower_bound"],
            markers["middle"]["upper_bound"],
        )
        xm_medium, ym_medium = find_color(middle_mask)
        # last end
        last_mask = cv2.inRange(
            hsv_frame,
            markers["last_end"]["lower_bound"],
            markers["last_end"]["upper_bound"],
        )
        xl_medium, yl_medium = find_color(last_mask)

        # middle to first
        cv2.line(frame, (xm_medium, ym_medium), (xf_medium, yf_medium), (255, 0, 0), 2)
        # middle to last
        cv2.line(frame, (xm_medium, ym_medium), (xl_medium, yl_medium), (255, 0, 0), 2)

        # middle angle
        angle = find_angle(
            (xf_medium, yf_medium), (xm_medium, ym_medium), (xl_medium, yl_medium)
        )

        cv2.putText(
            frame,
            str(angle) + " deg",
            (xm_medium, ym_medium),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_4,
        )
        cv2.imshow("Frame", frame)
        cv2.imshow("mask", first_mask + middle_mask + last_mask)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
