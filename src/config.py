class Config:

    def __init__(self, clothes='blouse'):
        # custom configs
        self.proj_path = 'D:/KPDA/fashion/'
        # self.proj_path = '/home/featurize/KPDA/fashion/'
        self.data_path = self.proj_path + 'KPDA/'
        self.batch_size_per_gpu = 6
        self.workers = 6
        self.gpus = '0,1'  # CUDA_DEVICES
        self.base_lr = 1e-3  # learning rate
        self.epochs = 100

        self.clothes = clothes
        self.keypoints = {
            'short_sleeved_shirt': ['key_point_'+str(i) for i in range(1, 26)],
            'long_sleeved_shirt': ['key_point_'+str(i) for i in range(26, 59)],
            'short_sleeved_outwear': ['key_point_'+str(i) for i in range(59, 90)],
            'long_sleeved_outwear': ['key_point_'+str(i) for i in range(90, 129)],
            'vest': ['key_point_'+str(i) for i in range(129, 144)],
            'sling': ['key_point_'+str(i) for i in range(144, 159)],
            'shorts': ['key_point_'+str(i) for i in range(159, 169)],
            'trousers': ['key_point_'+str(i) for i in range(169, 183)],
            'skirt': ['key_point_'+str(i) for i in range(183, 191)],
            'short_sleeved_dress': ['key_point_'+str(i) for i in range(191, 220)],
            'long_sleeved_dress': ['key_point_'+str(i) for i in range(220, 257)],
            'vest_dress': ['key_point_'+str(i) for i in range(257, 276)],
            'sling_dress': ['key_point_'+str(i) for i in range(276, 295)]
        }
        keypoint = self.keypoints[self.clothes]
        self.num_keypoints = len(keypoint)
        self.conjug = []
        for i, key in enumerate(keypoint):
            if 'left' in key:
                j = keypoint.index(key.replace('left', 'right'))
                self.conjug.append([i, j])
        if self.clothes in ['outwear', 'blouse', 'dress']:
            self.datum = [keypoint.index('armpit_left'), keypoint.index('armpit_right')]
        elif self.clothes in ['trousers', 'skirt']:
            self.datum = [keypoint.index('waistband_left'), keypoint.index('waistband_right')]
        # Img
        self.img_max_size = 512
        self.mu = 0.65
        self.sigma = 0.25
        # RPN
        # self.anchor_areas = [32 * 32., 64 * 64., 128 * 128., 256 * 256., 512 * 512.]  # p3 -> p7
        # self.aspect_ratios = [1 / 5., 1 / 2., 1 / 1., 2 / 1., 5 / 1.]  # w/h
        # self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        # self.anchor_num = len(self.aspect_ratios) * len(self.scale_ratios)
        self.max_iou = 0.7
        self.min_iou = 0.4
        self.cls_thresh = 0.5
        self.nms_thresh = 0.5
        self.nms_topk = 1

        # STAGE 2
        self.hm_stride = 4
        self.hm_sigma = self.img_max_size / self.hm_stride / 16.  # 4 #16 for 256 size
        self.hm_alpha = 100.

        # lrschedule = {'blouse': [16, 26, 42],
        #               'outwear': [15, 20, 26],
        #               'trousers': [18, 25, 36],
        #               'skirt': [26, 32, 39],
        #               'dress': [30, 34, 31]
        #               }
        # self.lrschedule = lrschedule[clothes]


if __name__ == '__main__':
    config = Config('dress')
    print(config.conjug)
