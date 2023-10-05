import numpy as np

class stereoCamera(object):
    def __init__(self):
        # 左右相机内参
        self.cam_matrix_left = np.transpose(np.array([[899.332729935937, 0, 0],
                                            [-0.0309415262783938, 895.190252023979, 0],
                                            [244.256039948844, 262.183374065655, 1]]))
        self.cam_matrix_right = np.transpose(np.array([[848.533054119787, 0, 0],
                                             [0.490920460884679, 845.009820475332, 0],
                                             [155.894008111804, 312.777872356747, 1]]))
        # 左右相机畸变系数 [k1,k2,p1,p2,k3]
        # k 径向畸变，p切向畸变
        self.distortion_l = np.array([-0.640173525235396, 2.22923482932109,
                                      -0.00186284065425854, 0.0124354447507034, -7.67146399917448])
        self.distortion_r = np.array([-0.367112629638545, -0.508379383614110,
                                      -0.00989295026141549, 0.0300020237169281, 1.23456548067471])
        # 旋转矩阵
        self.R = np.transpose(np.array([[0.998443517545486, -0.00776937329095990, -0.0552284266484483],
                                        [0.0114014048018427, 0.997770585746176, 0.0657561114139042],
                                        [0.0545944158311254, -0.0662834448289992, 0.996306155105778]]))
        # 平移矩阵
        self.T = np.array([119.252406828092, 1.37064775324791, -32.1510130574767])
        # 主点坐标的差 (principle point)
        self.doffs = 312.777872356747 - 262.183374065655
        # 指示内外参数是否为立体校正的结果
        self.isRectified = False

    # no use for setMiddleBurryParams
    """    def setMiddleBurryParams(self):

        self.cam_matrix_left = np.array([])
        self.cam_matrix_right = np.array([])
        self.distortion_l = np.zeros(shape=(5, 1), dyte=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dyte=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([])
        self.doffs = 0
        self.isRectified = True"""
