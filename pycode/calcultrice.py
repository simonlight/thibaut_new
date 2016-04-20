def every_fold_map(filepath):
    ap_dict = collections.defaultdict(lambda:collections.defaultdict(lambda:[]))
    with open(filepath) as apfile:
        for line in apfile:
            category, scale, index, ap_test = line.strip().split()
            ap_dict[scale.split(":")[1]][index.split(":")[1]].append(float(ap_test.split(":")[1]))
    for scale in ['50']:
        for index in ['0','1','2','3','4']:
            print index, np.mean(ap_dict[scale][index])
#             print ap_dict[scale]

if __name__ == "__main__":
    import collections
    import numpy as np
    print "act BB"
    every_fold_map("/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpBB_positive_cv_5fold/ap_summary.txt")
    print "act glsvm"
    every_fold_map("/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv_5fold_allscale_random_init_finaltest/ap_summary_ecarttype_seed1_detail.txt")
    print "obj BB"
    every_fold_map("/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpBB_positive_cv_5fold/ap_summary.txt")
    print "obj glsvm"
    every_fold_map("/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_cv_5fold_allscale_random_init_finaltest/ap_summary_ecarttype_seed1_detail.txt")

