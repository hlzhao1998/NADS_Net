from enum import IntEnum

from pathlib import Path

class JointType(IntEnum):
    """関節の種類を表す """
    Nose = 0
    """ 鼻 """
    Neck = 1
    """ 首 """
    RightShoulder = 2
    """ 右肩 """
    RightElbow = 3
    """ 右肘 """
    RightHand = 4
    """ 右手 """
    LeftShoulder = 5
    """ 左肩 """
    LeftElbow = 6
    """ 左肘 """
    LeftHand = 7
    """ 左手 """
    RightWaist = 8
    """ 右腰 """
    RightKnee = 9
    """ 右膝 """
    RightFoot = 10
    """ 右足 """
    LeftWaist = 11
    """ 左腰 """
    LeftKnee = 12
    """ 左膝 """
    LeftFoot = 13
    """ 左足 """
    RightEye = 14
    """ 右目 """
    LeftEye = 15
    """ 左目 """
    RightEar = 16
    """ 右耳 """
    LeftEar = 17
    """ 左耳 """

params = {
    'coco_dir': '/home/alanschen/dataset/coco2017/',
    'work_space': Path('work_space'),
    'log_path': 'work_space/log',

    'lr': 1e-4,
    # worker number mainly depends on your cpu core num
    'num_workers': 8,
    'batch_size': 8,

    # 'eva_num': 100,
    'eva_num': 1,
    # board 100 times in each epoch
    'board_loss_interval': 100,
    'eval_interval': 8,
    'board_pred_image_interval': 2,
    'save_interval': 8,

    'min_keypoints': 5,
    'min_area': 32 * 32,
    'insize': 368,
    'downscale': 8,
    'paf_sigma': 8,
    'heatmap_sigma': 7,
    
    'min_box_size': 64,
    'max_box_size': 512,
    'min_scale': 0.5,
    'max_scale': 2.0,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,

    # inference params
    'inference_img_size': 368,
    'inference_scales': [0.5, 1, 1.5, 2],
    # 'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,
    'n_integ_points_thresh': 8,
    'heatmap_peak_thresh': 0.05,
    'inner_product_thresh': 0.05,
    'limb_length_ratio': 1.0,
    'length_penalty_value': 1,
    'n_subset_limbs_thresh': 3,
    'subset_score_thresh': 0.2,
    'limbs_point': [
        [JointType.Neck, JointType.RightWaist],
        [JointType.RightWaist, JointType.RightKnee],
        [JointType.RightKnee, JointType.RightFoot],
        [JointType.Neck, JointType.LeftWaist],
        [JointType.LeftWaist, JointType.LeftKnee],
        [JointType.LeftKnee, JointType.LeftFoot],
        [JointType.Neck, JointType.RightShoulder],
        [JointType.RightShoulder, JointType.RightElbow],
        [JointType.RightElbow, JointType.RightHand],
        [JointType.RightShoulder, JointType.RightEar],
        [JointType.Neck, JointType.LeftShoulder],
        [JointType.LeftShoulder, JointType.LeftElbow],
        [JointType.LeftElbow, JointType.LeftHand],
        [JointType.LeftShoulder, JointType.LeftEar],
        [JointType.Neck, JointType.Nose],
        [JointType.Nose, JointType.RightEye],
        [JointType.Nose, JointType.LeftEye],
        [JointType.RightEye, JointType.RightEar],
        [JointType.LeftEye, JointType.LeftEar]
    ],
    'coco_joint_indices': [
        JointType.Nose,
        JointType.LeftEye,
        JointType.RightEye,
        JointType.LeftEar,
        JointType.RightEar,
        JointType.LeftShoulder,
        JointType.RightShoulder,
        JointType.LeftElbow,
        JointType.RightElbow,
        JointType.LeftHand,
        JointType.RightHand,
        JointType.LeftWaist,
        JointType.RightWaist,
        JointType.LeftKnee,
        JointType.RightKnee,
        JointType.LeftFoot,
        JointType.RightFoot
    ]
}
