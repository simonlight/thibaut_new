def read_ap(f):
    res=collections.defaultdict(lambda:[])
    for line in f:
        if line!="\n":
            category, scale,index,ap_test = line.strip().split()
            res[index.split(':')[1]].append(float(ap_test.split(':')[1]))
    return res

def pair_t_test(std,g):
    pair=[]
    for k in ['0','1','2','3','4']:
#         print "pair diff:%f"%(np.mean(g[k])-np.mean(std[k]))
        pair.append(np.mean(g[k])-np.mean(std[k]))
#         print np.mean(g[k]), np.mean(std[k])
    print "t-value:%f, mean:%f, std:%f"%(math.sqrt(5)*np.mean(pair)/np.std(pair),np.mean(pair), np.std(pair)) 


import collections
import numpy as np
import math
stdobjfile=open("/home/wangxin/tt/std_obj")
stdactfile=open("/home/wangxin/tt/std_act")
stdobjfile=open("/home/wangxin/tt/std_obj_randominit.txt")
stdactfile=open("/home/wangxin/tt/std_act_randominit.txt")

std_act_init0_1000 = read_ap(open("/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_scale30_init0_maxCCCP1000/ap_summary_ecarttype_seed1_detail.txt"))
std_obj_init0_1000 = read_ap(open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_scale30_init0_maxCCCP1000/ap_summary_ecarttype_seed1_detail.txt"))

obj100file=open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_standard_scale100_rigid_split/ap_summary_ecarttype_seed1_detail.txt")
act100file=open("/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_standard_scale100_rigid_split/ap_summary_ecarttype_seed1_detail.txt")
gobjfile=open("/home/wangxin/tt/g_obj")
gactfile=open("/home/wangxin/tt/g_act")

gobjrandomfile=open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_5fold_scale30_tradeoff0.2_random_init/ap_summary.txt")
gactrandomfile=open("/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_5fold_scale30_tradeoff0.2_random_init/ap_summary.txt")




gobjrandom=read_ap(gobjrandomfile)
gactrandom=read_ap(gactrandomfile)
std_obj=read_ap(stdobjfile)
std_act=read_ap(stdactfile)
g_obj=read_ap(gobjfile)
g_act=read_ap(gactfile)
act_100=read_ap(act100file)
obj_100=read_ap(obj100file)

# pair_t_test(std_obj, gobjrandom)
# pair_t_test(std_act, gactrandom)

# pair_t_test(std_obj,g_obj)
# pair_t_test(std_act,g_act)
pair_t_test(std_act_init0_1000, g_act)
pair_t_test(std_obj_init0_1000, g_obj)

