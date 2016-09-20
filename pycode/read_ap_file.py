def glsvm():
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


def npglsvm():
    import numpy as np
    scale_obj_list=[100,90,80,70,60,50,40,30]
    pos_tradeoff_obj_list=[0.1]
    neg_tradeoff_obj_list=[0.0,0.001]
    index_list=[0,1,2,3,4]
    for scale_obj in scale_obj_list:
        for pt in pos_tradeoff_obj_list:
            for nt in neg_tradeoff_obj_list:
                ap_test_all=[]
                for index_obj in index_list:
                    ap_test_split=0
                    f= open("/local/wangxin/results/upmc_food/npglsvm_food_traintrainlist_testtestlist_5split/ap_summary_ecarttype_seed1_detail.txt")
                    c=0
                    for line_index, line in enumerate(f):
                        category, lbd, scale, ptradeoff,ntradeoff, index, ap_test, ap_train = line.split()
                        if (int(scale.split(':')[1])==scale_obj and
                            float(ptradeoff.split(':')[1])==pt and 
                            float(ntradeoff.split(':')[1])==nt and
                            int(index.split(':')[1])==index_obj):
                            
                            c+=1
                            ap_test_split+=float(ap_test.split(':')[1])
                    f.close()
                    ap_test_all.append(ap_test_split/20)
                    
                print pt,nt,ap_test_all
#                 print "%4.2f,"%(np.std(ap_test_all)*100),
npglsvm()   
0.0,0.0,   74.98, 77.16, 79.26, 79.60, 79.24, 77.47, 75.22, 70.05,
0.0,0.001, 74.98, 77.13, 79.20, 79.26, 78.96, 77.44, 75.49, 69.68,
0.0,0.01,  74.98, 77.39, 79.57, 79.69, 78.89, 77.25, 75.13, 70.16,
0.0,0.1,   74.98, 77.73, 79.09, 78.48, 75.35, 73.98, 72.26, 65.55,

0.1,0.0,   74.98, 77.15, 79.57, 79.79, 79.26, 77.91, 75.76, 70.77,
1.06, 1.45, 1.39, 1.41, 0.74, 1.38, 1.54, 2.57,

0.1,0.001, 74.98, 77.21, 79.61, 79.93, 79.12, 77.89, 76.27, 71.75,
1.06, 1.28, 1.31, 1.62, 0.74, 1.43, 1.24, 1.87,

0.1,0.01,  74.98, 77.52, 79.78, 79.92, 79.34, 77.56, 75.90, 70.94,
0.1,0.1,   74.98, 77.57, 79.36, 78.81, 75.85, 74.86, 73.51, 66.70,


0.2,0.0,   74.98, 77.46, 79.62, 79.90, 79.48, 78.50, 75.71, 71.50,
0.2,0.001, 74.98, 77.53, 79.73, 79.97, 79.49, 78.60, 75.92, 71.65,
0.2, 0.01, 74.98, 77.58, 79.81, 80.19, 79.31, 78.01, 75.88, 70.97,
0.2,0.1,   74.98, 77.82, 79.23, 78.97, 76.42, 75.50, 73.86, 67.88,


0.5,0.0, 74.98, 77.69, 79.78, 79.85, 78.87, 77.93, 75.25, 70.86,
0.5,0.001, 74.98, 77.77, 79.73, 79.73, 78.87, 77.72, 74.66, 71.42,
0.5,0.01, 74.98, 77.78, 79.81, 80.17, 79.13, 77.96, 75.03, 70.13,
0.5,0.1, 74.98, 77.61, 79.11, 78.63, 76.47, 75.76, 73.73, 66.89,



    
    
    
                