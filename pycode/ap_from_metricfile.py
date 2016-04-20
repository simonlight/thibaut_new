def get_100x(f, feature_dict,label_dict):
    for line in f:
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature_dict[filename][0] = float(score)
        label_dict[filename] = yi
        
def get_score(f, scale, score_dict, label_dict):
       
    for line in f:
        score, yp,yi,hp,filename =  line.strip().split(',')
        score_dict[filename] = float(score)
        label_dict[filename] = yi

def getAP(label_value_list):
    label_value_list = sorted(label_value_list, key = lambda x: x[1], reverse = True)
    example_num = len(label_value_list)
    tp = np.zeros(example_num, dtype= np.float)
    fp = np.zeros(example_num, dtype= np.float)

    cumtp = 0
    cumfp = 0 
    totalpos = 0
    
    for cnt, e in enumerate(label_value_list):
        if e[0] == 1:
            cumtp+=1
            totalpos+=1
        else:
            cumfp+=1
        tp[cnt] = cumtp;
        fp[cnt] = cumfp;

    prec = np.zeros(example_num, dtype= np.float)
    reca = np.zeros(example_num, dtype= np.float)
    
    for i in range(example_num):
        reca[i] = tp[i]/totalpos
        prec[i] = tp[i]/(tp[i]+fp[i])
    
    mrec = np.zeros(example_num+2, dtype= np.float)
    for cnt, r in enumerate(reca):
        mrec[cnt+1] = r
    mrec[-1] = 1
    
    mpre = np.zeros(example_num+2, dtype= np.float)
    for cnt, p in enumerate(prec):
        mpre[cnt+1] = p

    for i in range(example_num, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    
    mAP = 0
    for j in range(1, example_num+1):
        mAP += (mrec[j]-mrec[j-1])*mpre[j]
    return mAP

if __name__=="__main__":
    import os.path as op
    import collections
    import numpy as np

    
    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv/metric_final/"
    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_standard/metric_final/"
    root = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_standard/metric_final/"
    root = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_cv/metric_final/"

    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_standard_cv_5fold_allscale/metric/"
    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv_5fold_allscale/metric/"
    

    root = "/local/wangxin/results/ferrari_gaze/std_et/final_test_30_extra_glsvm/metric/"
    root = "/local/wangxin/results/ferrari_gaze/std_et/final_test_30_extra_lsvm/metric/"
    root = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_standard_cv_5fold_allscale_random_init_finaltest/metric/"
    root = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_cv_5fold_allscale_random_init_finaltest/metric/"
    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv_5fold_allscale_random_init_finaltest/metric/"
    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_standard_cv_5fold_allscale_random_init_finaltest/metric/"
    
    categories = ["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    categories = ["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]

# lsvm_cccpgaze_positive_cv
    splitAP=collections.defaultdict(None)
    for split in [0,1,2,3,4]:
        print "split:%d"%split
        
        scaleAP=collections.defaultdict(None)
        for scale in ['100','30','40','50','60','70','80','90']:
            category_ap=[]
            for category in categories:
#             for scale in ['100','60','70','80','90']:
                score_dict= collections.defaultdict(None)
                label_dict= collections.defaultdict(None)
                
#                 f = open(op.join(root, '_'.join(['metric','val',scale,'0.2','0.0_1.0E-4',category,str(split)+'.txt'])))
                f = open(op.join(root, '_'.join(['metric','val',scale,'0.0_1.0E-4',category,str(split)+'.txt'])))
#                 f = open(op.join(root, scale,'_'.join(['metric','valval',scale,'0.0_1.0E-4',category+'.txt'])))
                get_score(f,scale, score_dict, label_dict)
                g = zip([int(e) for e in label_dict.values()],score_dict.values())
                category_ap.append(getAP(g))
                if scale == "90":
                    print category, getAP(g)
            
            scaleAP[scale]=np.mean(category_ap)
        splitAP[split]=scaleAP
    
    for scale in ['100','30','40','50','60','70','80','90']:
        temp_list=[]
        for key in [0,1,2,3,4]:
            temp_list.append(splitAP[key][scale])
#             if scale == "30":
            print splitAP[key][scale]
        print scale, np.mean(temp_list)*100,np.std(temp_list)*100 
    #     for test_X in test_feature_dict.values():
            

