import cv2
import argparse
from NADS_Net_Solver import NADS_Net_Solver, draw_person_pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--weights', help='weights file path')
    parser.add_argument('--img', help='image file path')
    args = parser.parse_args()

    # load model
    nads_net = NADS_Net_Solver(weights_file = args.weights, training = False)

    # read image
    img = cv2.imread(args.img)

    # inference
    poses, _ = nads_net.detect(img)

    # draw and save image
    img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)

    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)