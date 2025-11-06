from ....stats_utils import DISTRIBUTIONS


# NOTE 1: 'distr_id' can be also a list with same shape as 'values' and 'std' to 
# NOTE 1: specify a specific distribution for each value in 'values'

# NOTE 2: 'std' can also be one float value. in this case all values get the 
# NOTE 2: same standard deviation


NNET10_PARAMS = {
    'layer_1': {
        'bias': {
            'values': [1.46807317643, 0.914303376435, 1.11779465009, -1.88774024593, -1.44203980969],
            'distr': {
                'distr_id': DISTRIBUTIONS.NORMAL_1D,
                'std': 0.05
            }
        },
        'affine_transform': {
            'values': [
                [0.631232653841,1.52145695394,-1.11728836958,0.0705656588538,-0.210100752689,-0.0509035777766],
                [-0.994197749515,1.35142800933,0.682282724537,0.805184200972,-0.137840226219,-0.602358856112],
                [0.22438211584,-0.0666837070779,-1.23975437825,-0.0155768528013,0.112222977626,0.0363639300463],
                [-0.966688976161,0.0999483658509,-0.313320370927,1.1063608678,0.739904923122,-1.48041296773],
                [-0.24339645819,-0.418138871993,-0.710773649761,0.811567335788,0.552429192687,1.17282293897]
            ],
            'distr': {
                'distr_id': DISTRIBUTIONS.NORMAL_1D,
                'std': 0.03
            }
        }
    },
    'layer_2': {
        'bias': {
            'values': [0.00472943029704],
            'distr': {
                'distr_id': DISTRIBUTIONS.GAMMA_1D,
                'std': 0.0005
            }
        },
        'affine_transform': {
            'values': [[-0.34462874064, 0.0341910410835, -0.737283566458, 0.00125809202955, 0.00395679366549]],
            'distr': {
                'distr_id': DISTRIBUTIONS.NORMAL_1D,
                'std': [[0.05, 0.005, 0.05, 0.0003, 0.0003]]
            }
        }
    }
}