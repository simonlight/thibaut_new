
def res_tradeoff_file_2_dict(td_folder):
    loss_tradeoff = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : lambda : collections.defaultdict(lambda : None))))
    loss = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    loss_baseline = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(lambda : None)))
    
    for fp in os.listdir(td_folder):
        category, scale, eps, lbd, tradeoff, _,_,_,_,_ =fp.strip().split('-')
        with open(os.path.join(td_folder, fp)) as f:
            iterations = f.readlines()[:-1]
            classification_loss, positive_gaze_loss_bound, negative_gaze_loss_bound, gaze_loss_bound,\
            positive_gaze_loss, negative_gaze_loss, gaze_loss, objectif = [line.strip().split(":")[1] for line in iterations]
            
            loss_tradeoff[scale][category][tradeoff]= [float(classification_loss), float(positive_gaze_loss_bound), float(negative_gaze_loss_bound),
                                                      float(gaze_loss_bound), float(positive_gaze_loss), float(negative_gaze_loss),
                                                      float(gaze_loss), float(objectif)]
    
    for k1 in res_tradeoff.keys():
        for k2 in res_tradeoff[k1].keys():
            max_val = -1
            max_tradeoff = -1
            for k3 in res_tradeoff[k1][k2].keys():
                if res_tradeoff[k1][k2][k3][1] > max_val:
                    
                    max_val = res_tradeoff[k1][k2][k3][1]
                    max_tradeoff = k3
            res[k1][k2] = res_tradeoff[k1][k2][max_tradeoff]
            res_baseline[k1][k2] = res_tradeoff[k1][k2]['0.0']
    return res, res_baseline
    
def get_y(ap_res, scale_cv):

    y_train = [0]*len(scale_cv)
    #     y_val = [0]*len(scale_cv)
    y_test = [0]*len(scale_cv)


    for scale in scale_cv:

        result_name = scale
        x_axis = ap_res1[scale].keys()
#         tradeoff_cv = [0.0,0.1,0.5,1.0,1.5,2.0,5.0,10.0]
        
        y_ap_all = ap_res[scale]
        print y_ap_all
#             print tradeoff, y_ap_all
#         print scale, tradeoff, np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())
#             print np.sum(y_ap_all.values(), axis=0)
        ap_train, ap_val, ap_test = np.sum(y_ap_all.values(), axis=0) / len(y_ap_all.values())

        y_train[scale_cv.index(scale)] = ap_train
#         y_val[scale_cv.index(scale)] = ap_val
        y_test[scale_cv.index(scale)] = ap_test
    return y_train, y_test

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
    
def plot_2_methods(ap_res1, ap_res2, res_typ):
    scale_cv = ['90','80','70','60', '50', '40', '30']
    cccpgaze_positive_train, cccpgaze_positive_test = get_y(ap_res1 ,scale_cv)
    cccp_train, cccp_test = get_y(ap_res2, scale_cv)

    plt.figure(figsize=(8,4))
    plt.plot(scale_cv,cccpgaze_positive_train,label="lsvm cccpgaze_positive train "+res_typ,color="red",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccpgaze_positive_test,label="lsvm cccpgaze_positive test "+res_typ,color="red",linewidth=2)
    
    plt.plot(scale_cv,cccp_train,label="lsvm cccp train "+res_typ,color="green",linewidth=2)
#     plt.plot(scale_cv,y_val,label="validation "+res_typ,color="blue",linewidth=2)
    plt.plot(scale_cv,cccp_test,label="lsvm cccp test "+res_typ,color="green",linewidth=2)    
    
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
    td_folder = "/local/wangxin/results/full_stefan_gaze/lsvm_et/lsvm_cccpgaze_positive_cv/trainingdetail"

    ap_res1,res_baseline = res_tradeoff_file_2_dict(td_folder)
#     ap_results = open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_cccp/ap_summary.txt")
#     ap_res2 = res_file_2_dict(ap_results)
#     ap_results = open("/local/wangxin/results/ferrari_gaze/std_et/lsvm_standard/ap_summary.txt")
#     ap_res3 = res_file_2_dict(ap_results)
#     plot_res(ap_res1, ap_res2, ap_res3, "AP")
    plot_2_methods(ap_res1, res_baseline,  "mAP")