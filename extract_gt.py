import os
import numpy as np
import pandas as pd

if __name__ == '__main__':

    GT = pd.read_csv('oxford_dataset/TownCentre-groundtruth.top', header=None)
    indent = lambda x, y: ''.join(['  ' for _ in range(y)]) + x

    factor = 2
    train_size = 4500
    xml_path = 'oxford_dataset/annotations'
    # os.mkdir(xml_path)
    name = 'pedestrian'
    width, height = 300, 300
    factor_x = 300 / 1920
    factor_y = 300 / 1080

    for frame_number in range(train_size):

        Frame = GT.loc[GT[1] == frame_number]
        x1 = list(Frame[8])
        y1 = list(Frame[11])
        x2 = list(Frame[10])
        y2 = list(Frame[9])
        points = [[(round(x1_), round(y1_)), (round(x2_), round(y2_))] for x1_, y1_, x2_, y2_ in zip(x1, y1, x2, y2)]

        with open(os.path.join(xml_path, 'annotations_' + str(frame_number) + '.xml'), 'w') as file:
            file.write('<annotation>\n')
            file.write(indent('<filename>' + 'frame_' + str(frame_number) + '.jpg' + '</filename>\n', 1))
            file.write(indent('<size>\n', 1))
            file.write(indent('<width>' + str(width) + '</width>\n', 2))
            file.write(indent('<height>' + str(height) + '</height>\n', 2))
            file.write(indent('<depth>3</depth>\n', 2))
            file.write(indent('</size>\n', 1))

            for point in points:

                top_left = point[0]
                bottom_right = point[1]

                if top_left[0] > bottom_right[0]:
                    xmax, xmin = top_left[0] * factor_x, bottom_right[0] * factor_x
                else:
                    xmin, xmax = top_left[0] * factor_x, bottom_right[0] * factor_x

                if top_left[1] > bottom_right[1]:
                    ymax, ymin = top_left[1] * factor_y, bottom_right[1] * factor_y
                else:
                    ymin, ymax = top_left[1] * factor_y, bottom_right[1] * factor_y

                file.write(indent('<object>\n', 1))
                file.write(indent('<name>' + name + '</name>\n', 2))
                file.write(indent('<bndbox>\n', 2))
                file.write(indent('<xmin>' + str(int(xmin)) + '</xmin>\n', 3))
                file.write(indent('<ymin>' + str(int(ymin)) + '</ymin>\n', 3))
                file.write(indent('<xmax>' + str(int(xmax)) + '</xmax>\n', 3))
                file.write(indent('<ymax>' + str(int(ymax)) + '</ymax>\n', 3))
                file.write(indent('</bndbox>\n', 2))
                file.write(indent('</object>\n', 1))

            file.write('</annotation>\n')

        print('File:', frame_number, end='\r')