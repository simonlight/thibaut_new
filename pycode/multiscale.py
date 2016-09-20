def get_100x(f, feature_dict,label_dict,scales):
    for cnt,line in enumerate(f):
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature_dict[cnt][scales.index("100")] = float(score)
        label_dict[cnt] = int(yi)
        
def get_x(f, feature_dict,scale, scales):
    for cnt,line in enumerate(f):
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature_dict[cnt][scales.index(scale)] = float(score)

def get_100x_multiclass(f, feature,label,scales,cat_cnt):
    for cnt,line in enumerate(f):
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature[cnt][scales.index("100")+len(scales)*cat_cnt] = float(score)
        label[cnt][cat_cnt]=int(yi)
        
def get_x_multiclass(f, feature,scale,scales,cat_cnt):
    for cnt,line in enumerate(f):
        score, yp,yi,hp,filename =  line.strip().split(',')
        feature[cnt][scales.index(scale)+len(scales)*cat_cnt] = float(score)

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

def acc_monoscale_binary_salienceprediction_multiclass(root_metric_folder, root_label_folder, categories, scales,types,verbose):
    #acc effect of scale
    for typ in types:
        if typ=="test":
            num_example=200
        for s in scales:
            res_list=[]
            scores=np.zeros([num_example, 20])
            labels=np.zeros([num_example, 20])
            for cat_cnt,c in enumerate(categories):
                f= open(root_metric_folder+'/'+s+'/'+c+'.txt')
                for ex_cnt,line in enumerate(f):
                    score=line.strip()
                    scores[ex_cnt][cat_cnt]=float(score)
                f.close()
                f= open(root_label_folder+'/'+c+'.txt')
                for ex_cnt,line in enumerate(f):
                    score0, score1= line.strip().split()
                    labels[ex_cnt][cat_cnt]=int(np.argmax([float(score0), float(score1)]))
                    
                f.close()
            print scores[0]
            pred_labels = np.argmax(scores,axis=1)
            print pred_labels[0]
            gt_labels = np.argmax(labels,axis=1)
            if verbose:
                print "splits:%s, typ:%s, scale:%s,  mean_acc:%f"%(0,typ, s, accuracy_score(gt_labels, pred_labels))
            print "%f,"%accuracy_score(gt_labels, pred_labels),
#                 res_list.append(accuracy_score(gt_labels, pred_labels))
#             print 

def acc_monoscale_binarymulticlass(root_metric_folder, categories, scales,tradeoff,types,splits,verbose):
    #acc effect of scale
    for typ in types:
        if typ=="train":
            num_example=1600
        if typ=="val":
            num_example=400
        for s in scales:
            for t in tradeoff:
                res_list=[]
                for sp in splits:
                    scores=np.zeros([num_example, 20])
                    labels=np.zeros([num_example, 20])
                    for cat_cnt,c in enumerate(categories):
                        f= open(root_metric_folder+'/'+ '_'.join(["metric",typ, s, t, "0.0_1.0E-4",c,sp+'.txt']))
                        for ex_cnt,line in enumerate(f):
                            score, yp,yi,hp,id=line.split(',')
                            scores[ex_cnt][cat_cnt]=float(score)
                            if yi=='1':
                                labels[ex_cnt][cat_cnt]=1
                    pred_labels = np.argmax(scores,axis=1)
                    gt_labels = np.argmax(labels,axis=1)
                    if verbose:
                        print "splits:%s, typ:%s, scale:%s, tradeoff:%s, mean_acc:%f"%(sp,typ, s, t, accuracy_score(gt_labels, pred_labels))
                    print "%f,"%accuracy_score(gt_labels, pred_labels),
                    res_list.append(accuracy_score(gt_labels, pred_labels))
                print 
#                 print "%4.2f,"%(np.std(res_list)*100),

def acc_multiscale_binarymulticlass(root, categories, scales, splits, tradeoffs,verbose, manuel,svm_c):
    for tradeoff in tradeoffs:
        for c in svm_c:
            all_split_acc=[]
            for split in splits:
                
                feature_train = np.zeros([1600,len(scales)*len(categories)])
                label_train = np.zeros([1600, len(categories)])    
                for cat_cnt,category in enumerate(categories):
                    for scale in scales:
                        if scale == '100':
                            f = open(op.join(root, '_'.join(['metric','train',scale,'0.0','0.0_1.0E-4',category,str(split)+'.txt'])))
        #                     f = open(op.join(root, '_'.join(['metric','train',scale,'0.0_1.0E-4',category,str(split)+'.txt'])))
                            get_100x_multiclass(f, feature_train, label_train, scales,cat_cnt)
                        elif manuel and scale=="70":
                                f = open(op.join(root, '_'.join(['metric','train',scale,tradeoff,'0.0_0.001',category,str(split)+'.txt'])))
                                get_x_multiclass(f, feature_train, scale, scales,cat_cnt)
                        elif manuel and scale=="80":
                                f = open(op.join(root, '_'.join(['metric','train',scale,tradeoff,'0.0_0.001',category,str(split)+'.txt'])))
                                get_x_multiclass(f, feature_train, scale, scales,cat_cnt)
                        else:  
                            f = open(op.join(root, '_'.join(['metric','train',scale,tradeoff, '0.0_1.0E-4',category,str(split)+'.txt'])))
        #                         f = open(op.join(root, '_'.join(['metric','train',scale, '0.0_1.0E-4',category,str(split)+'.txt'])))
                            get_x_multiclass(f, feature_train,scale,scales,cat_cnt)
                label_train=np.argmax(label_train,axis=1)
                from sklearn.svm import LinearSVC
                clf = LinearSVC(C=c)
                
                clf.fit(feature_train, label_train) 
        
                feature_test = np.zeros([400,len(scales)*len(categories)])
                
                label_test = np.zeros([400,len(categories)])               
                for cat_cnt,category in enumerate(categories):
                    for scale in scales:
                        for cible_tradeoff in tradeoffs:
                            if scale == '100':
                                f = open(op.join(root, '_'.join(['metric','val',scale,'0.0','0.0_1.0E-4',category,str(split)+'.txt'])))
                                get_100x_multiclass(f, feature_test, label_test, scales,cat_cnt)
                            elif manuel and scale=="70":
                                f = open(op.join(root, '_'.join(['metric','val',scale,tradeoff,'0.0_0.001',category,str(split)+'.txt'])))
                                get_x_multiclass(f, feature_test, scale, scales,cat_cnt)
                            elif manuel and scale=="80":
                                f = open(op.join(root, '_'.join(['metric','val',scale,tradeoff,'0.0_0.001',category,str(split)+'.txt'])))
                                get_x_multiclass(f, feature_test, scale, scales,cat_cnt)            
                            else:  
                                f = open(op.join(root, '_'.join(['metric','val',scale, tradeoff,'0.0_1.0E-4',category,str(split)+'.txt'])))
                                get_x_multiclass(f, feature_test,scale, scales,cat_cnt)
                label_test=np.argmax(label_test,axis=1)
                if verbose:
                    print "split:%s, train_acc:%f"%(split,accuracy_score(clf.predict(feature_train),label_train))
                    print "split:%s, test_acc:%f"%(split,accuracy_score(clf.predict(feature_test),label_test))
                all_split_acc.append(accuracy_score(clf.predict(feature_test),label_test))
                print accuracy_score(clf.predict(feature_test),label_test)
            print "tradeoff:%s, c:%f, average acc:%f, std:%f"%(tradeoff, c, np.mean(all_split_acc), np.std(all_split_acc))


def ap_multiscale_binaryclass(root, categories, scales,tradeoffs,splits,verbose, manuel,svm_c):
    training_example=1600
    test_example=400
    for tradeoff in tradeoffs:
        for c in svm_c:
            average_ap=[]
            for split in splits:
                total_ap=[]
                for cat_cnt,category in enumerate(categories):
                    feature_train = np.zeros([training_example,len(scales)])
                    label_train = np.zeros([training_example])   
                    for scale in scales:
                        if scale == '100':
                            f = open(op.join(root, '_'.join(['metric','train',scale,'0.0','0.0_1.0E-4',category,str(split)+'.txt'])))
                            get_100x(f, feature_train, label_train, scales)
                        elif manuel and scale=="70":
                            
                            f = open(op.join(root, '_'.join(['metric','train',scale,'0.0','0.0_0.001',category,str(split)+'.txt'])))
                            get_x(f, feature_train, scale, scales)
                        elif manuel and scale=="80":
                            f = open(op.join(root, '_'.join(['metric','train',scale,'0.0','0.0_0.001',category,str(split)+'.txt'])))
                            get_x(f, feature_train, scale, scales)    
                        else:  
                            f = open(op.join(root, '_'.join(['metric','train',scale,tradeoff, '0.0_1.0E-4',category,str(split)+'.txt'])))
                            get_x(f, feature_train,scale, scales)
                    from sklearn.svm import LinearSVC
                    clf = LinearSVC(C=c)
                
                    clf.fit(feature_train, label_train) 
        
                    feature_test = np.zeros([test_example,len(scales)])
                    label_test = np.zeros([test_example])                
                
                    for scale in scales:
        #             for scale in ['100','60','70','80','90']:
                        if scale == '100':
                            f = open(op.join(root, '_'.join(['metric','val',scale,'0.0','0.0_1.0E-4',category,str(split)+'.txt'])))
        #                     f = open(op.join(root, '_'.join(['metric','val',scale,'0.0_1.0E-4',category,str(split)+'.txt'])))
                            get_100x(f, feature_test, label_test, scales)
                        elif manuel and scale=="70":
                            f = open(op.join(root, '_'.join(['metric','val',scale,'0.0','0.0_0.001',category,str(split)+'.txt'])))
                            get_x(f, feature_train, scale, scales)     
                        elif manuel and scale=="80":
                            f = open(op.join(root, '_'.join(['metric','val',scale,'0.0','0.0_0.001',category,str(split)+'.txt'])))
                            get_x(f, feature_train, scale, scales)        
                        else:  
                            f = open(op.join(root, '_'.join(['metric','val',scale, tradeoff,'0.0_1.0E-4',category,str(split)+'.txt'])))
        #                         f = open(op.join(root, '_'.join(['metric','val',scale,'0.0_1.0E-4',category,str(split)+'.txt'])))
                            get_x(f, feature_test,scale, scales)
                
                    g = zip([int(e) for e in label_test],clf.decision_function(feature_test))
                
                    
                    total_ap.append(getAP(g))
                average_ap.append(total_ap)
                if verbose:
                    print "split:%s, avg ap:%f"%(split, sum(total_ap)/len(categories))
            classwise_mean_split=np.mean(average_ap, axis=1)
            print "tradeoff:%s, c:%f, average ap:%f, std:%f"%(tradeoff, c, np.mean(classwise_mean_split), np.std(classwise_mean_split))

if __name__=="__main__":
    import os.path as op
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
    root = "/local/wangxin/results/upmc_food/glsvm_food_traintrainlist_testtestlist_80/full_metric"
    root = "/local/wangxin/results/upmc_food/glsvm_food_traintrainlist_testtestlist_5split/metric"
    categories = ["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
    categories = ["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    
    categories = ["apple-pie","bread-pudding","beef-carpaccio","beet-salad",
                      "chocolate-cake","chocolate-mousse","donuts","beignets",
                      "eggs-benedict","croque-madame","gnocchi","shrimp-and-grits",
                      "grilled-salmon","pork-chop","lasagna","ravioli",
                      "pancakes","french-toast","spaghetti-bolognese","pad-thai"]
    scales=["90"]
    splits=['0','1','2','3','4']
    verbose = False
    manuel = False

    tradeoffs=["0.0"]
    svm_c=[0.1]
#     ap_multiscale_binaryclass(root, categories, scales,tradeoffs,splits,verbose,manuel,svm_c)
    types=["test"]
#     acc_monoscale_binarymulticlass(root, categories, scales, tradeoffs, types,splits, verbose)
#     acc_multiscale_binarymulticlass(root, categories, scales, splits, tradeoffs,verbose, manuel,svm_c)
    metric_root = "/local/wangxin/results/upmc_food/tf_mlp_gaze_reg_classwise_clf_binary"
    label_root="/local/wangxin/results/upmc_food/tf_mlp_gaze_reg_classwise_clf_binary/labels"
    acc_monoscale_binary_salienceprediction_multiclass(metric_root, label_root, categories, scales,types,verbose)


    