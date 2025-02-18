import numpy as np


class Dist:

    # num_res = 2 number of resources in the system
    # max_job_size = 10  maximum resource request of new work
    # max_job_len = 15  maximum duration of new job
    def __init__(self, num_res, max_nw_size, job_len):
        self.num_res = num_res
        self.max_nw_size = max_nw_size # 10
        self.job_len = job_len

        self.job_small_chance = 0.9

        self.job_len_big_lower = job_len * 2 / 3 # 10
        self.job_len_big_upper = job_len         # 15

        self.job_len_small_lower = 1             # 1
        self.job_len_small_upper = job_len / 5   # 3

        self.dominant_res_lower = max_nw_size / 2 # 5
        self.dominant_res_upper = max_nw_size     # 10

        self.other_res_lower = 1                  # 1
        self.other_res_upper = max_nw_size / 5    # 2

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self):

        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)  # 1..3
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)    # 10..15

        nw_size = np.zeros(self.num_res)

        # -- job resource request --

        assert(self.num_res == 2) # 2 resources
        # job can have memory of 512*(1,2,3,4,5,6), up to 3GB
        # CPU is proportional to memory

        mem_resource = np.random.randint(1,6+1) # number between 1 and 6
        cpu_resource = mem_resource
        nw_size[0] = mem_resource
        nw_size[1] = cpu_resource

        #dominant_res = np.random.randint(0, self.num_res)
        #for i in range(self.num_res):
        #    if i == dominant_res:
        #        nw_size[i] = np.random.randint(self.dominant_res_lower,
        #                                       self.dominant_res_upper + 1) # 5..10
        #    else:
        #        nw_size[i] = np.random.randint(self.other_res_lower,
        #                                       self.other_res_upper + 1) # 1..2

        return nw_len, nw_size


def generate_sequence_work(pa, seed=42):

    np.random.seed(seed)

    simu_len = pa.simu_len * pa.num_ex

    nw_dist = pa.dist.bi_model_dist

    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)

    for i in range(simu_len):

        if np.random.rand() < pa.new_job_rate:  # a new job comes

            nw_len_seq[i], nw_size_seq[i, :] = nw_dist()

    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq,
                             [pa.num_ex, pa.simu_len, pa.num_res])

    return nw_len_seq, nw_size_seq
