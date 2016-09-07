def get_100x(f, feature,label,cat_cnt):
    for cnt,line in enumerate(f):
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature[cnt][0+8*cat_cnt] = float(score)
        label[cnt][cat_cnt]=int(yi)
        
def get_x(f, feature,scale,cat_cnt):
    index = ((100-int(scale))/10)
    
    for cnt,line in enumerate(f):
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature[cnt][index+8*cat_cnt] = float(score)
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
    feature_train = np.zeros([1600,8*20])
    label_train = np.zeros([1600, 20])               
    ap=[]
    cible_tradeoff="0.0"
    for split in [0]:
        average_ap=0
        average_acc=0
        for cat_cnt,category in enumerate(categories):
            for scale in ['100','30','40','50','60','70','80','90']:
#             for scale in ['100','30']:
                if scale == '100':
                    f = open(op.join(root, '_'.join(['metric','train',scale,'0.0','0.0_1.0E-4',category,str(split)+'.txt'])))
#                     f = open(op.join(root, '_'.join(['metric','train',scale,'0.0_1.0E-4',category,str(split)+'.txt'])))
                    get_100x(f, feature_train, label_train,cat_cnt)
                elif scale=="70":
                    for tradeoff in [cible_tradeoff]:
                        f = open(op.join(root, '_'.join(['metric','train',scale,'0.0','0.0_0.001',category,str(split)+'.txt'])))
                        get_x(f, feature_train, scale,cat_cnt) 
                else:  
                    for tradeoff in [cible_tradeoff]:


                        f = open(op.join(root, '_'.join(['metric','train',scale,tradeoff, '0.0_1.0E-4',category,str(split)+'.txt'])))
#                         f = open(op.join(root, '_'.join(['metric','train',scale, '0.0_1.0E-4',category,str(split)+'.txt'])))
                        get_x(f, feature_train,scale,cat_cnt)
#         feature_dict = L2_norm(feature_dict)
#             print np.shape(label_dict.values())
#             print np.shape(feature_dict.values())
        label_train=np.argmax(label_train,axis=1)
        print label_train
        from sklearn import svm
        for c in [0.001,0.1,0.1,1]:
            clf = svm.SVC(C=c)
            
            clf.fit(feature_train, label_train) 
            print accuracy_score(clf.predict(feature_train),label_train)
    
            feature_test = np.zeros([200,8*20])
            
            label_test = np.zeros([200,20])               
            for cat_cnt,category in enumerate(categories):
                for scale in ['100','30','40','50','60','70','80','90']:
                    if scale == '100':
                        f = open(op.join(root, '_'.join(['metric','val',scale,'0.0','0.0_1.0E-4',category,str(split)+'.txt'])))
                        get_100x(f, feature_test, label_test,cat_cnt)
                    elif scale=="70":
                        for tradeoff in [cible_tradeoff]:
                            f = open(op.join(root, '_'.join(['metric','val',scale,'0.0','0.0_0.001',category,str(split)+'.txt'])))
                            get_x(f, feature_train, scale,cat_cnt)            
                    else:  
                        for tradeoff in [cible_tradeoff]:
                            f = open(op.join(root, '_'.join(['metric','val',scale, tradeoff,'0.0_1.0E-4',category,str(split)+'.txt'])))
                            get_x(f, feature_test,scale,cat_cnt)
                    
            label_test=np.argmax(label_test,axis=1)
            print accuracy_score(clf.predict(feature_test),label_test)
                    
         
             
        
            
         
    