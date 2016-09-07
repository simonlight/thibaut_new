def get_100x(f, feature_dict,label_dict):
    for cnt,line in enumerate(f):
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature_dict[cnt][0] = float(score)
        label_dict[cnt] = int(yi)
        
def get_x(f, feature_dict,scale):
    index = ((100-int(scale))/10)
    
    for cnt,line in enumerate(f):
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature_dict[cnt][index] = float(score)
import math
    
def L2_norm(feature_dict):
    for k,v in feature_dict.items():
        fea = feature_dict[k]
        l2 = math.sqrt(sum([d*d for d in fea]))
        feature_dict[k] = [d/l2 for d in fea]
        return feature_dict

import numpy as np

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
    from sklearn.metrics import accuracy_score

    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv/metric_final/"
    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_standard/metric_final/"
    root = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_standard/metric_final/"
    root = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_cv/metric_final/"

    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_standard_cv_5fold_allscale/metric/"
    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv_5fold_allscale/metric/"

    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_standard_cv_5fold_allscale_random_init_finaltest/metric/"
    root = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_standard_cv_5fold_allscale_random_init_finaltest/metric/"
    root = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv_5fold_allscale_random_init_finaltest/metric/"
    
    root = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_cv_5fold_allscale_random_init_finaltest/metric/"
    root = "/local/wangxin/results/upmc_food/glsvm_food_traintrainlist_testtestlist_70/full_metric"
    categories = ["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    categories = ["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    
    categories = [
            "apple-pie",
            "bread-pudding",
            "beef-carpaccio",
            "beet-salad",
            "chocolate-cake",
            "chocolate-mousse",
            "donuts",
            "beignets",
            "eggs-benedict",
            "croque-madame",
            "gnocchi",
            "shrimp-and-grits",
            "grilled-salmon",
            "pork-chop",
            "lasagna",
            "ravioli",
            "pancakes",
            "french-toast",
            "spaghetti-bolognese",
            "pad-thai"        
            ]
    
# lsvm_cccpgaze_positive_cv
    total_ap=[]
    cible_tradeoff="0.1"
    for split in [0]:
        average_ap=0
        average_acc=0
        for cat_cnt,category in enumerate(categories):
            feature_train = np.zeros([1600,8])
            label_train = np.zeros([1600])   

            for scale in ['100','30','40','50','60','70','80','90']:
#             for scale in ['100','30']:
                if scale == '100':
                    f = open(op.join(root, '_'.join(['metric','train',scale,'0.0','0.0_1.0E-4',category,str(split)+'.txt'])))
#                     f = open(op.join(root, '_'.join(['metric','train',scale,'0.0_1.0E-4',category,str(split)+'.txt'])))
                    get_100x(f, feature_train, label_train)
                elif scale=="70":
                    for tradeoff in [cible_tradeoff]:
                        f = open(op.join(root, '_'.join(['metric','train',scale,'0.0','0.0_0.001',category,str(split)+'.txt'])))
                        get_x(f, feature_train, scale)     
                else:  
                    for tradeoff in [cible_tradeoff]:
                        f = open(op.join(root, '_'.join(['metric','train',scale,tradeoff, '0.0_1.0E-4',category,str(split)+'.txt'])))
                        get_x(f, feature_train,scale)
            from sklearn import svm
            clf = svm.SVC(C=0.001)
        
            clf.fit(feature_train, label_train) 

        
            scores=[]
            labels=[]
            
            feature_test = np.zeros([200,8])
            label_test = np.zeros([200])                
        
            for scale in ['100','30','40','50','60','70','80','90']:
#             for scale in ['100','60','70','80','90']:
                if scale == '100':
                    f = open(op.join(root, '_'.join(['metric','val',scale,'0.0','0.0_1.0E-4',category,str(split)+'.txt'])))
#                     f = open(op.join(root, '_'.join(['metric','val',scale,'0.0_1.0E-4',category,str(split)+'.txt'])))
                    get_100x(f, feature_test, label_test)
                elif scale=="70":
                    for tradeoff in [cible_tradeoff]:
                        f = open(op.join(root, '_'.join(['metric','val',scale,'0.0','0.0_0.001',category,str(split)+'.txt'])))
                        get_x(f, feature_train, scale)     
                       
                else:  
                    for tradeoff in [cible_tradeoff]:
                        f = open(op.join(root, '_'.join(['metric','val',scale, tradeoff,'0.0_1.0E-4',category,str(split)+'.txt'])))
#                         f = open(op.join(root, '_'.join(['metric','val',scale,'0.0_1.0E-4',category,str(split)+'.txt'])))
                        get_x(f, feature_test,scale)
        
            g = zip([int(e) for e in label_test],clf.decision_function(feature_test))
        
            
            total_ap.append(getAP(g))
        average_ap =sum(total_ap)/20
        
    print "average ap: "+str(average_ap)
        
            
         
    