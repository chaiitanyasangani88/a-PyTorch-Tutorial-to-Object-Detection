import cv2


def extract_frames_from_video(path):
    vc = cv2.VideoCapture(path)
    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    frame_num = 0
    success = 1

    while success:

        success, image = vc.read()
        if success:
            cv2.imwrite(f"oxford_dataset/extracted_images_1/frame_{frame_num}.jpg", image)

            frame_num += 1
            if frame_num % 500 == 0:
                print(f'frame {frame_num}')
        else:
            break


if __name__ == "__main__":
    extract_frames_from_video('oxford_dataset/TownCentreXVID.avi')