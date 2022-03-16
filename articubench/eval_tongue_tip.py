import torch
import pandas as pd

from .control_models import control_models_to_evaluate


# halt = halten; ja = jaja
SEMVECS = {'ja': torch.tensor(
      [-0.0737, -0.0786, -0.0415, -0.0277,  0.0313, -0.0467,  0.0048,
       -0.0126, -0.0881,  0.0715,  0.0427, -0.0844,  0.0361, -0.0612,
       -0.0325,  0.0214, -0.0014,  0.1223,  0.1085, -0.0517,  0.0024,
        0.024 , -0.0046,  0.0247, -0.0598,  0.0472, -0.1918,  0.0464,
       -0.1821,  0.0448,  0.0347, -0.0334, -0.0154, -0.0054,  0.0614,
        0.0908, -0.0034,  0.0215, -0.0294, -0.0392, -0.0913, -0.003 ,
        0.09  ,  0.0333, -0.0827,  0.2377, -0.0033,  0.007 , -0.055 ,
        0.0262, -0.0083, -0.0376, -0.0118, -0.0779, -0.0057,  0.0419,
        0.0295, -0.0631, -0.0822,  0.0448, -0.011 , -0.0692,  0.0604,
        0.0302, -0.0154, -0.0345,  0.0753, -0.1035,  0.1256, -0.0554,
        0.0823, -0.0442,  0.1294, -0.0194, -0.1307, -0.1001, -0.05  ,
       -0.0551,  0.0472, -0.0942,  0.0055,  0.0786,  0.0641,  0.1086,
       -0.0783,  0.0787,  0.0714, -0.0062,  0.0597, -0.0398,  0.1045,
        0.0253,  0.0082, -0.0364, -0.0436,  0.0577, -0.0465,  0.0032,
        0.0075,  0.0182,  0.0459,  0.1053,  0.0356, -0.1293, -0.0154,
       -0.072 ,  0.0061, -0.0945, -0.0727,  0.0059, -0.0007,  0.1114,
       -0.0526,  0.02  ,  0.0487, -0.016 ,  0.0315, -0.0382,  0.0039,
       -0.0676,  0.0638, -0.0443,  0.0887,  0.0429,  0.1194, -0.0089,
       -0.0365, -0.0865,  0.0442,  0.0499,  0.0403,  0.1029,  0.017 ,
        0.0053,  0.1027,  0.0675,  0.0071,  0.0585,  0.05  ,  0.0423,
        0.1228, -0.1826,  0.0726, -0.0521, -0.0322,  0.0914, -0.0836,
        0.0321, -0.0321,  0.0294,  0.0773,  0.0049, -0.0087, -0.0868,
        0.1084,  0.0313, -0.0497, -0.0645,  0.0094,  0.0876, -0.0409,
       -0.0392,  0.0685, -0.0407,  0.0677,  0.054 , -0.0049, -0.0304,
        0.0512,  0.0357, -0.1196, -0.0493, -0.0167,  0.06  ,  0.0215,
        0.0048, -0.0145, -0.1237, -0.1963,  0.0032,  0.0894, -0.0074,
        0.005 , -0.1399, -0.0548,  0.1139,  0.024 ,  0.0174, -0.0716,
       -0.0173,  0.0449,  0.0147, -0.0268, -0.0268, -0.124 ,  0.0731,
        0.0352,  0.1432,  0.0227,  0.0028, -0.0849, -0.0558,  0.0029,
       -0.0661,  0.0041,  0.0229,  0.0022, -0.0576,  0.0227,  0.0531,
       -0.1176,  0.0295,  0.0088,  0.0991,  0.0135, -0.0213, -0.0474,
        0.0176,  0.0686, -0.0923,  0.0429,  0.0246, -0.0174, -0.0366,
        0.0262,  0.0238,  0.0194,  0.0789, -0.0833,  0.0115,  0.1031,
        0.0756, -0.0066, -0.0282,  0.0442,  0.0052,  0.0091,  0.0897,
        0.1656,  0.0025,  0.0122,  0.0935, -0.0092, -0.122 , -0.037 ,
        0.0651, -0.0382, -0.0071, -0.0412, -0.0524,  0.0759,  0.0863,
       -0.0842,  0.0518,  0.076 ,  0.0778,  0.027 ,  0.0225, -0.0171,
        0.0641,  0.0906, -0.0582, -0.1051,  0.0601,  0.0194,  0.0027,
        0.004 , -0.0727,  0.1203,  0.0851,  0.0791, -0.0175, -0.1261,
        0.0162, -0.1134, -0.0141,  0.1519,  0.009 , -0.0468,  0.0095,
       -0.0605, -0.0139,  0.1224,  0.0065,  0.0846,  0.0272, -0.0056,
       -0.037 ,  0.0114, -0.0105, -0.0287,  0.0101, -0.0015,  0.0447,
        0.0138, -0.0328,  0.0545,  0.0798, -0.0389,  0.0547]).double(),
        'halt': torch.tensor(
       [2.700e-03,  1.900e-02, -1.810e-02,  1.070e-02,  2.500e-03,
       -7.020e-02,  6.480e-02,  2.370e-02, -2.230e-02,  1.440e-02,
        1.101e-01, -3.720e-02, -1.290e-02, -6.770e-02, -5.800e-02,
       -2.510e-02,  5.850e-02, -2.410e-02,  1.550e-02, -3.280e-02,
        2.100e-02, -4.020e-02,  3.970e-02, -6.000e-03, -6.000e-04,
       -1.250e-02, -1.700e-03, -1.960e-02,  1.530e-02, -7.370e-02,
        8.230e-02, -1.270e-02, -3.880e-02, -8.300e-03,  3.550e-02,
       -9.900e-03, -2.990e-02, -2.230e-02,  9.830e-02,  4.370e-02,
       -1.650e-02, -7.000e-03,  1.880e-02,  2.070e-02, -6.500e-03,
        7.560e-02, -5.100e-03, -3.170e-02,  7.340e-02, -3.400e-02,
       -5.600e-02,  6.800e-03,  3.400e-03, -4.230e-02, -2.900e-02,
        1.100e-03,  2.790e-02,  6.210e-02, -6.710e-02, -3.580e-02,
       -3.830e-02, -7.100e-02, -5.440e-02,  2.430e-02,  8.800e-03,
       -3.300e-03,  3.270e-02, -3.630e-02,  1.470e-02,  4.900e-02,
        8.450e-02,  5.830e-02,  7.800e-03,  3.140e-02, -5.660e-02,
       -1.100e-02,  2.580e-02,  1.180e-02,  4.500e-03, -4.000e-03,
       -1.100e-02,  8.320e-02, -3.780e-02,  1.244e-01, -4.000e-02,
        4.260e-02,  1.900e-03, -1.760e-02,  3.420e-02,  0.000e+00,
        6.450e-02,  9.200e-03, -3.370e-02,  5.510e-02,  3.420e-02,
        2.020e-02,  1.300e-03,  8.000e-04,  7.260e-02,  4.270e-02,
       -1.100e-03, -4.200e-03,  2.600e-02, -2.900e-02,  2.330e-02,
       -9.710e-02,  5.940e-02,  1.220e-02,  4.780e-02,  4.230e-02,
        6.310e-02, -5.500e-03, -3.310e-02,  7.800e-02,  1.490e-02,
       -6.830e-02, -7.500e-03,  6.700e-03,  7.990e-02, -5.890e-02,
       -4.960e-02,  3.640e-02,  8.200e-03, -2.700e-03, -8.500e-03,
       -3.360e-02, -6.340e-02, -2.860e-02,  1.100e-03,  2.000e-02,
        2.170e-02, -8.600e-03,  3.360e-02,  3.800e-03, -2.300e-03,
        2.160e-02, -1.810e-02,  7.140e-02,  6.750e-02, -3.790e-02,
        6.490e-02, -1.125e-01, -3.290e-02,  4.910e-02,  1.850e-02,
        3.790e-02,  5.300e-02, -7.140e-02,  2.850e-02,  1.000e-04,
        4.460e-02, -1.330e-02, -1.410e-02, -7.270e-02, -1.100e-03,
        8.400e-03, -3.220e-02,  2.700e-03,  2.710e-02,  6.520e-02,
        1.530e-02,  4.200e-03,  2.350e-02, -6.000e-03,  1.410e-02,
       -3.290e-02,  1.420e-02, -6.800e-03,  2.360e-02, -8.700e-03,
        4.030e-02,  1.570e-02, -2.160e-02, -2.600e-02, -4.200e-03,
        1.056e-01,  3.080e-02,  3.260e-02, -3.550e-02, -7.520e-02,
        4.830e-02,  4.240e-02,  3.140e-02, -4.080e-02, -4.930e-02,
        1.400e-03, -1.250e-02, -2.870e-02,  3.910e-02,  3.670e-02,
       -3.900e-03, -1.320e-02,  1.600e-03,  3.100e-03,  2.200e-03,
        5.800e-02,  5.650e-02,  1.560e-02, -3.230e-02,  6.140e-02,
       -3.280e-02, -7.450e-02, -1.830e-02,  5.930e-02,  1.440e-02,
       -1.090e-02,  8.700e-03, -9.100e-03, -4.750e-02,  2.960e-02,
        4.560e-02, -5.070e-02,  5.140e-02,  1.260e-02, -1.220e-02,
       -7.550e-02, -8.480e-02,  2.180e-02, -9.780e-02,  3.070e-02,
        2.910e-02, -1.500e-02, -3.700e-03,  1.000e-02, -1.260e-01,
       -1.620e-02,  1.120e-02,  1.910e-02, -2.400e-03, -6.900e-03,
       -6.800e-03,  5.000e-04,  1.360e-02,  2.260e-02,  4.790e-02,
       -1.140e-02, -5.240e-02,  4.380e-02,  6.750e-02,  2.730e-02,
        3.690e-02,  2.850e-02,  1.060e-02, -8.110e-02,  1.690e-02,
        3.430e-02,  1.094e-01,  2.430e-02, -6.900e-03, -1.504e-01,
        3.600e-03,  5.300e-03,  9.460e-02, -6.990e-02, -4.000e-03,
        7.290e-02, -9.500e-03, -9.600e-03,  3.940e-02,  8.400e-03,
        5.460e-02, -2.240e-02, -2.540e-02,  1.470e-02,  6.190e-02,
        1.660e-02, -5.230e-02,  1.390e-02, -4.970e-02,  7.500e-02,
        2.720e-02,  3.760e-02, -7.820e-02,  4.170e-02,  5.000e-02,
       -2.650e-02,  6.510e-02,  2.140e-02,  5.300e-03, -8.880e-02,
       -5.990e-02,  1.840e-02,  6.820e-02,  2.110e-02, -3.790e-02,
       -1.140e-02,  5.000e-03, -7.530e-02, -4.330e-02,  2.290e-02,
        2.830e-02,  1.870e-02, -4.330e-02,  5.190e-02,  9.100e-03,
       -4.090e-02,  1.490e-02,  7.980e-02, -8.100e-03, -1.330e-02]).double()}

KEC_JA_HALT_DATA = pd.read_pickle('data/KEC/ja_halt.pickle')

def tonge_tip_tiny():

    # /halt/
    halt = KEC_JA_HALT_DATA.iloc[55]
    # /ja/
    ja = KEC_JA_HALT_DATA.iloc[16]

    for name, model in control_models_to_evaluate.items():
        print(name + " /ja/")
        # synthesize
        n_samples = int(ja.Signal.shape[0] / 110 * (44100 / ja.SampleRate))
        cps = model(n_samples,
                target_semantic_vector=SEMVECS['ja'],
                target_audio=ja.Signal,
                sampling_rate=ja.SampleRate)

        # create EMA

# TODO: NOT FINISHED, look at score.py first
# TODO: continue here: /home/tino/Documents/phd/projects/essv2020/04_apply_measures.py

