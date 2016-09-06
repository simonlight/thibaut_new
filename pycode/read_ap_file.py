scale_obj_list=[100]
tradeoff_obj_list=[0.0]
for scale_obj in scale_obj_list:
    for tradeoff_obj in tradeoff_obj_list:
        f= open("/local/wangxin/results/upmc_food/glsvm_food_traintrainlist_testtestlist/ap_summary_ecarttype_seed1_detail.txt")
        ap_test_all=0
        c=0
        for line in f:
            category, lbd, scale, tradeoff, index, ap_test, ap_train = line.split()
            if int(scale.split(':')[1])==scale_obj and float(tradeoff.split(':')[1])==tradeoff_obj:
                c+=1
                ap_test_all+=float(ap_test.split(':')[1])
#         print scale_obj, tradeoff_obj, c, ap_test_all/c
        print str(ap_test_all/c) + ',',
        f.close()
    print 
        