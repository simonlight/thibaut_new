def res_file_2_dict(ap_results):
    res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    for line in ap_results:
        category, scale, lbd, epsilon, ap_train, ap_val, ap_test = [i.split(":")[1] for i in line.strip().split()]
        res[scale][category]= [float(ap_train), float(ap_val), float(ap_test)]
    return res

def res_tradeoff_file_2_dict(ap_results, td_folder):
    res_tradeoff = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : lambda : collections.defaultdict(lambda : None))))
    res = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    res_baseline = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    max_tradeoff_dict= collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    
    for line in ap_results:
        category, tradeoff, scale, lbd, epsilon, ap_train, ap_val, ap_test = [i.split(":")[1] for i in line.strip().split()]
        res_tradeoff[scale][category][tradeoff]= [float(ap_train), float(ap_val), float(ap_test)]
    
    for k1 in res_tradeoff.keys():
        for k2 in res_tradeoff[k1].keys():
            max_val = -1
            max_tradeoff = -1
            for k3 in res_tradeoff[k1][k2].keys():
                if res_tradeoff[k1][k2][k3][1] > max_val:
                    max_val = res_tradeoff[k1][k2][k3][1]
                    max_tradeoff = k3
            res[k1][k2] = res_tradeoff[k1][k2][max_tradeoff]
            max_tradeoff_dict[k1][k2] = max_tradeoff
            res_baseline[k1][k2] = res_tradeoff[k1][k2]['0.0']
    #loss    
    loss_tradeoff = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : lambda : collections.defaultdict(lambda : None))))
    loss = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    loss_baseline = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    
    for fp in os.listdir(td_folder):
        category, scale, eps, lbd, tradeoff, _,_,_,_,_ =fp.strip().split('_')
        with open(os.path.join(td_folder, fp)) as f:
            iterations = f.readlines()[:-1]
            for line in iterations:
                elements = line.strip().split()
                classification_loss, positive_gaze_loss_bound, negative_gaze_loss_bound, gaze_loss_bound,\
                positive_gaze_loss, negative_gaze_loss, gaze_loss, objectif = [ele.split(":")[1] for ele in elements]
            
            loss_tradeoff[scale][category][tradeoff]= [float(classification_loss), float(positive_gaze_loss_bound), float(negative_gaze_loss_bound),
                                                      float(gaze_loss_bound), float(positive_gaze_loss), float(negative_gaze_loss),
                                                      float(gaze_loss), float(objectif)]

    for k1 in res_tradeoff.keys():
        for k2 in res_tradeoff[k1].keys():
            loss[k1][k2] = loss_tradeoff[k1][k2][max_tradeoff_dict[k1][k2]]
            loss_baseline[k1][k2] = loss_tradeoff[k1][k2]['0.0']
    
    
    return res, res_baseline, res_tradeoff, loss, loss_baseline
    
def get_y(ap_res, scale_cv):

    y_train = [0]*len(scale_cv)
    #     y_val = [0]*len(scale_cv)
    y_test = [0]*len(scale_cv)


    for scale in scale_cv:

        y_ap_all = ap_res[scale]

        ap_train, ap_val, ap_test = np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())

        y_train[scale_cv.index(scale)] = ap_train
        y_test[scale_cv.index(scale)] = ap_test
    return y_train, y_test

def get_loss(ap_res, scale_cv):
    d1 = [0]*len(scale_cv)
    d2 = [0]*len(scale_cv)
    d3 = [0]*len(scale_cv)
    d4 = [0]*len(scale_cv)
    d5 = [0]*len(scale_cv)
    d6 = [0]*len(scale_cv)
    d7 = [0]*len(scale_cv)
    d8 = [0]*len(scale_cv)


    for scale in scale_cv:

        result_name = scale
        x_axis = ap_res1[scale].keys()
#         tradeoff_cv = [0.0,0.1,0.5,1.0,1.5,2.0,5.0,10.0]
        
        y_ap_all = ap_res[scale]
#         print y_ap_all
#             print tradeoff, y_ap_all
#         print scale, tradeoff, np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())
#             print np.sum(y_ap_all.values(), axis=0)
        classification_loss, positive_gaze_loss_bound, negative_gaze_loss_bound, gaze_loss_bound,\
            positive_gaze_loss, negative_gaze_loss, gaze_loss, objectif = np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())

        d1[scale_cv.index(scale)] = classification_loss
        d2[scale_cv.index(scale)] = positive_gaze_loss_bound
        d3[scale_cv.index(scale)] = negative_gaze_loss_bound
        d4[scale_cv.index(scale)] = gaze_loss_bound
        d5[scale_cv.index(scale)] = positive_gaze_loss
        d6[scale_cv.index(scale)] = negative_gaze_loss
        d7[scale_cv.index(scale)] = gaze_loss
        d8[scale_cv.index(scale)] = objectif

    return d1, d2, d3, d4, d5, d6, d7,d8


def plot_2_loss(loss_res1, loss_baseline,  res_typ):
    scale_cv = ['90','80','70','60', '50', '40', '30']
    cccp_positive_d1, cccp_positive_d2, cccp_positive_d3, cccp_positive_d4, cccp_positive_d5, cccp_positive_d6, cccp_positive_d7,cccp_positive_d8 = get_loss(loss_res1 ,scale_cv)
    cccplsvm_d1, cccplsvm_d2, cccplsvm_d3, cccplsvm_d4, cccplsvm_d5, cccplsvm_d6, cccplsvm_d7,cccplsvm_d8 = get_loss(loss_baseline ,scale_cv)

    plt.figure(figsize=(8,4))
    plt.plot(scale_cv,cccp_positive_d1,label="cccpgaze_positive cls loss",color="red",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccp_positive_d5,label="cccpgaze_positive positive loss ",color="blue",linewidth=2)
#     plt.plot(scale_cv,cccp_positive_d6,label="cccpgaze_+ negative loss ",color="green",linewidth=2)
#     plt.plot(scale_cv,cccp_positive_d7,label="cccpgaze_+ loss ",color="black",linewidth=2)

    plt.plot(scale_cv,cccplsvm_d1,label="cccp cls loss ",color="red",linewidth=2, linestyle = "dashed")
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccplsvm_d5,label="cccp positive loss ",color="blue",linewidth=2, linestyle = "dashed")
#     plt.plot(scale_cv,cccplsvm_d6,label="cccp negative loss ",color="green",linewidth=2, linestyle = "dashed")
#     plt.plot(scale_cv,cccplsvm_d7,label="cccp loss ",color="black",linewidth=2, linestyle = "dashed")


    plt.xlabel("Scale")
    plt.ylabel(res_typ)
#     plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
#     plt.title(res_typ+" of scale:%s, gain of test:%4.2f %%"%(scale,(y_test[y_val.index(max(y_val))]-y_test[0])*100))
    plt.title("action loss")
#     plt.axvline(x=scale_cv[y_val.index(max(y_val))], color= 'black', linestyle='dashed')
    plt.legend(loc='best',fancybox=True,framealpha=0.5)
    plt.show()

def plot_1_methods(ap_res1, res_typ):
    scale_cv = ['90','80','70','60', '50', '40', '30']
    cccpgaze_positive_train, cccpgaze_positive_test = get_y(ap_res1 ,scale_cv)
    plt.figure(figsize=(8,4))
    plt.plot(scale_cv,cccpgaze_positive_train,label="lsvm cccpgaze_positive train "+res_typ,color="red",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccpgaze_positive_test,label="lsvm cccpgaze_positive test "+res_typ,color="red",linewidth=2)

    plt.xlabel("Scale")
    plt.ylabel(res_typ)
#     plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
#     plt.title(res_typ+" of scale:%s, gain of test:%4.2f %%"%(scale,(y_test[y_val.index(max(y_val))]-y_test[0])*100))
    plt.title("train/val examples")
#     plt.axvline(x=scale_cv[y_val.index(max(y_val))], color= 'black', linestyle='dashed')
    plt.legend(loc='best',fancybox=True,framealpha=0.5)
    plt.show()


def plot_every_tradeoff_2(ap_res1, ap_baseline, res_typ):
    scale_cv = ['90','80','70','60', '50', '40', '30']
    cccpgaze_positive_train = [0]*len(scale_cv)
    #     y_val = [0]*len(scale_cv)
    cccpgaze_positive_test = [0]*len(scale_cv)
    
    lsvm_cccp_train, lsvm_cccp_test = get_y(ap_baseline, scale_cv)
    
    tradeoff_cv = [0.0, '1.0E-4',0.001,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
    for gamma in tradeoff_cv:
        for scale in scale_cv:

            y_ap_all = ap_res1[scale].values()
            print str(gamma)
            y_ap_all = [y[str(gamma)] for y in y_ap_all]
            print y_ap_all
            ap_train, ap_val, ap_test = np.sum(y_ap_all, axis=0) / len(y_ap_all)
    
            cccpgaze_positive_train[scale_cv.index(scale)] = ap_train
            cccpgaze_positive_test[scale_cv.index(scale)] = ap_test
        
        plt.figure(figsize=(8,4))

        plt.plot(scale_cv,cccpgaze_positive_train, label="lsvm cccpgaze_positive train "+res_typ,color="red",linewidth=2)
        plt.plot(scale_cv,cccpgaze_positive_test, label="lsvm cccpgaze_positive test "+res_typ,color="red",linewidth=2)
        
        plt.plot(scale_cv,lsvm_cccp_train,label="lsvm cccp train "+res_typ,color="green",linewidth=2)
        plt.plot(scale_cv,lsvm_cccp_test,label="lsvm cccp test "+res_typ,color="green",linewidth=2)
        
        
        plt.xlabel("Scale")
        plt.ylabel(res_typ)
    #     plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
    #     plt.title(res_typ+" of scale:%s, gain of test:%4.2f %%"%(scale,(y_test[y_val.index(max(y_val))]-y_test[0])*100))
        plt.title("gamma=%s, action classification: train/test results"%str(gamma))
    #     plt.axvline(x=scale_cv[y_val.index(max(y_val))], color= 'black', linestyle='dashed')
        plt.legend(loc='best',fancybox=True,framealpha=0.5)
        plt.show()   

def plot_every_tradeoff_3(ap_res1, ap_baseline, ap_lsvm_standard, res_typ):
    scale_cv = ['90','80','70','60', '50', '40', '30']
    cccpgaze_positive_train = [0]*len(scale_cv)
    #     y_val = [0]*len(scale_cv)
    cccpgaze_positive_test = [0]*len(scale_cv)
    
    lsvm_cccp_train, lsvm_cccp_test = get_y(ap_baseline, scale_cv)
    
    lsvm_standard_train, lsvm_standard_test = get_y (ap_lsvm_standard, scale_cv)
    
    tradeoff_cv = [0.0, '1.0E-4',0.001,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
    for gamma in tradeoff_cv:
        for scale in scale_cv:

            y_ap_all = ap_res1[scale].values()
            print str(gamma)
            y_ap_all = [y[str(gamma)] for y in y_ap_all]
            print y_ap_all
    #             print tradeoff, y_ap_all
    #         print scale, tradeoff, np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())
    #             print np.sum(y_ap_all.values(), axis=0)
            ap_train, ap_val, ap_test = np.sum(y_ap_all, axis=0) / len(y_ap_all)
    
            cccpgaze_positive_train[scale_cv.index(scale)] = ap_train
    #         y_val[scale_cv.index(scale)] = ap_val
            cccpgaze_positive_test[scale_cv.index(scale)] = ap_test
        
        
        
        plt.figure(figsize=(8,4))
        plt.plot(scale_cv,cccpgaze_positive_train, label="lsvm cccpgaze_positive train "+res_typ,color="red",linewidth=2)
    #     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
        plt.plot(scale_cv,cccpgaze_positive_test, label="lsvm cccpgaze_positive test "+res_typ,color="red",linewidth=2)
        
        plt.plot(scale_cv,lsvm_cccp_train,label="lsvm cccp train "+res_typ,color="green",linewidth=2)
        #     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
        plt.plot(scale_cv,lsvm_cccp_test,label="lsvm cccp test "+res_typ,color="green",linewidth=2)
        
        plt.plot(scale_cv,lsvm_standard_train,label="lsvm standard train "+res_typ,color="blue",linewidth=2)
        #     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
        plt.plot(scale_cv,lsvm_standard_test,label="lsvm standard test "+res_typ,color="blue",linewidth=2)    
    
        
        plt.xlabel("Scale")
        plt.ylabel(res_typ)
    #     plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
    #     plt.title(res_typ+" of scale:%s, gain of test:%4.2f %%"%(scale,(y_test[y_val.index(max(y_val))]-y_test[0])*100))
        plt.title("gamma=%s, object classification: train/test results"%str(gamma))
    #     plt.axvline(x=scale_cv[y_val.index(max(y_val))], color= 'black', linestyle='dashed')
        plt.legend(loc='best',fancybox=True,framealpha=0.5)
        plt.show()

def plot_2_methods(ap_res1, ap_res2, res_typ):
    scale_cv = ['90','80','70','60', '50', '40', '30']
    cccpgaze_positive_train, cccpgaze_positive_test = get_y(ap_res1 ,scale_cv)
    print "####"

    cccp_train, cccp_test = get_y(ap_res2, scale_cv)

    plt.figure(figsize=(8,4))
    plt.plot(scale_cv,cccpgaze_positive_train, label="lsvm cccpgaze_positive train "+res_typ,color="red",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccpgaze_positive_test, label="lsvm cccpgaze_positive test "+res_typ,color="red",linewidth=2)
    
    plt.plot(scale_cv,cccp_train, label="lsvm cccp train "+res_typ,color="green",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccp_test, label="lsvm cccp test "+res_typ,color="green",linewidth=2)    
    
    plt.xlabel("Scale")
    plt.ylabel(res_typ)
#     plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
#     plt.title(res_typ+" of scale:%s, gain of test:%4.2f %%"%(scale,(y_test[y_val.index(max(y_val))]-y_test[0])*100))
    plt.title("action classification: train/test results")
#     plt.axvline(x=scale_cv[y_val.index(max(y_val))], color= 'black', linestyle='dashed')
    plt.legend(loc='best',fancybox=True,framealpha=0.5)
    plt.show()
    
def plot_3_methods(ap_res1, ap_res2, ap_res3, res_typ):
    scale_cv = ['90','80','70','60', '50', '40', '30']

    cccpgaze_positive_train, cccpgaze_positive_test = get_y(ap_res1 ,scale_cv)
    cccp_train, cccp_test = get_y(ap_res2, scale_cv)
    standard_train, standard_test = get_y(ap_res3, scale_cv)
        
    plt.figure(figsize=(8,4))
    plt.plot(scale_cv,cccpgaze_positive_train,label="lsvm cccpgaze_positive train "+res_typ,color="red",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccpgaze_positive_test,label="lsvm cccpgaze_positive test "+res_typ,color="red",linewidth=2)
    
    plt.plot(scale_cv,cccp_train,label="lsvm cccp train "+res_typ,color="green",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccp_test,label="lsvm cccp test "+res_typ,color="green",linewidth=2)    
    
    plt.plot(scale_cv,standard_train,label="lsvm train "+res_typ,color="blue",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,standard_test,label="lsvm test "+res_typ,color="blue",linewidth=2)    
    
    plt.xlabel("Scale")
    plt.ylabel(res_typ)
#     plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
#     plt.title(res_typ+" of scale:%s, gain of test:%4.2f %%"%(scale,(y_test[y_val.index(max(y_val))]-y_test[0])*100))
    plt.title("train/val examples")
#     plt.axvline(x=scale_cv[y_val.index(max(y_val))], color= 'black', linestyle='dashed')
    plt.legend(loc='best',fancybox=True,framealpha=0.5)
    plt.show()



if __name__=='__main__':
    import collections
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    ap_results = open("/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv/ap_summary.txt")
    td_folder = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv/trainingdetail"

#     ap_results = open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_cv/ap_summary.txt")
#     td_folder = "/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccpgaze_positive_cv/trainingdetail"
    
    ap_res1, ap_baseline, res_tradeoff, loss_res1, loss_baseline = res_tradeoff_file_2_dict(ap_results, td_folder)
#     ap_results = open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccp/ap_summary.txt")
#     ap_res2 = res_file_2_dict(ap_results)
#     ap_lsvm_standard_f = open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_standard/ap_summary.txt")
#     ap_lsvm_standard = res_file_2_dict(ap_lsvm_standard_f)
#     plot_res(ap_res1, ap_res2, ap_res3, "AP")

#     plot_2_methods(ap_res1, ap_baseline,  "mAP")
    plot_every_tradeoff_2(res_tradeoff, ap_baseline,"mAP")
#     plot_2_loss(loss_res1, loss_baseline,  "loss")