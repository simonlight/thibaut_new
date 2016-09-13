import numpy as np
scale_obj_list=[100,90,80,70,60,50,40,30]
tradeoff_obj_list=[0.2]
index_list=[0,1,2,3,4]
for scale_obj in scale_obj_list:
    for tradeoff_obj in tradeoff_obj_list:
        ap_test_all=[]
        for index_obj in index_list:
            ap_test_split=0
            f= open("/local/wangxin/results/upmc_food/glsvm_food_traintrainlist_testtestlist_5split/ap_summary_ecarttype_seed1_detail.txt")
            c=0
            for line in f:
                category, lbd, scale, tradeoff, index, ap_test, ap_train = line.split()
                if int(scale.split(':')[1])==scale_obj and float(tradeoff.split(':')[1])==tradeoff_obj and int(index.split(':')[1])==index_obj:
                    c+=1
                    ap_test_split+=float(ap_test.split(':')[1])
            f.close()
            ap_test_all.append(ap_test_split/20)
        print "%4.2f,"%(np.mean(ap_test_all)*100),






            