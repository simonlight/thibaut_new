import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)  
scale_cv = ['100',  '90', '80','70', '60', '50', '40','30']

#obj
lsvm_mean=[88.2,89.24,89.63,89.5,89.2,88.8,87.2,84.18]
lsvm_std=[1.2,1.15,1.2,1.11,0.99,.87,1.13,0.52]
glsvm_mean=[88.2,89.27,89.68,89.62,89.53,88.95,87.73,85.13]
glsvm_std=[1.2,1.17,1.2,1.1,.87,1.02,1.13,.49]
#act
lsvm_mean=[60.8, 62.6,63.8,64.9,64.3,63.8,62.4,58.4]
lsvm_std=[1.2, 1.0,1.2,1.3,1.2,0.4,1.0,1.6]
glsvm_mean=[60.8, 62.8,64.0,65.4,65.5,65.6,65.24,62.16]
glsvm_std=[1.2, 0.8,1.0,1.1,1.1,.75,1.16,1.06]
#food acc multiclass/monoscale
lsvm_mean_00=[71.85, 72.35, 75.25, 76.05, 74.95, 73.65, 71.45, 67.20,]
glsvm_mean_01=[71.85, 72.05, 75.30, 75.60, 75.90, 74.50, 72.15, 67.95,]
glsvm_mean_02=[71.85, 72.35, 76.50, 75.45, 76.00, 74.75, 73.15, 68.35,]
glsvm_mean_05=[71.85, 72.60, 76.35, 75.45, 74.55, 72.95, 71.70, 66.85,]
glsvm_mean_10=[71.85, 72.85, 76.10, 74.35, 73.65, 71.55, 67.25, 63.20,]
lsvm_std_00=[1.87, 1.52, 2.02, 1.24, 1.94, 1.36, 1.04, 1.09,]
glsvm_std_01=[1.87, 1.54, 1.81, 1.35, 1.38, 0.89, 1.02, 1.44,]
glsvm_std_02=[1.87, 1.85, 1.47, 1.77, 1.62, 0.76, 1.47, 1.62,]
glsvm_std_05=[1.87, 1.93, 1.33, 0.58, 1.25, 0.89, 0.81, 0.96,]
glsvm_std_10=[1.87, 1.62, 1.87, 0.87, 1.59, 1.20, 1.04, 0.86,]

#food ap binaryclass/monoscale
lsvm_mean_00=[75.08, 77.09, 79.38, 79.64, 78.57, 77.53, 75.00, 70.02,]
glsvm_mean_01=[75.08, 77.31, 79.50, 79.93, 79.17, 77.75, 75.76, 71.09,]
glsvm_mean_02=[75.08, 77.49, 79.85, 79.89, 79.15, 78.09, 75.91, 71.15,]
glsvm_mean_05=[75.08, 77.76, 79.85, 79.96, 78.97, 77.68, 75.29, 71.09,]
glsvm_mean_10=[75.08, 77.70, 79.42, 79.01, 77.62, 75.76, 71.82, 66.79,]

lsvm_std_00=[1.09, 1.26, 1.31, 1.29, 0.75, 1.89, 1.37, 1.51,]
glsvm_std_01=[1.09, 1.30, 1.28, 1.63, 0.92, 1.19, 1.05, 1.78,]
glsvm_std_02=[1.09, 1.44, 1.38, 1.47, 0.89, 1.35, 1.38, 1.89,]
glsvm_std_05=[1.09, 1.51, 1.15, 1.35, 0.73, 1.12, 1.49, 2.02,]
glsvm_std_10=[1.09, 1.73, 1.17, 1.65, 1.35, 1.00, 1.13, 1.62,]

t_test=True
if t_test:
    import numpy as np
    import math
    def pair_t_test(baseline,new_method):
        pair = np.array(new_method)-np.array(baseline)
        print  "t-value:%f, mean:%f, std:%f"%(math.sqrt(5)*np.mean(pair)/np.std(pair),np.mean(pair), np.std(pair))
    baseline_40=[0.707500, 0.720000, 0.707500, 0.732500, 0.705000,]
    new_method_40=[0.722500, 0.747500, 0.725000, 0.750000, 0.712500,]
    baseline_50=[0.750000, 0.747500, 0.720000, 0.745000, 0.720000,]
    new_method_50=[0.757500, 0.740000, 0.750000, 0.752500, 0.737500,]
    baseline_60=[0.732500, 0.765000, 0.780000, 0.735000, 0.735000,]
    new_method_60=[0.765000, 0.787500, 0.745000, 0.760000, 0.742500,]
    baseline_70=[0.767500, 0.770000, 0.770000, 0.757500, 0.737500,]
    new_method_70=[0.745000, 0.775000, 0.752500, 0.772500, 0.727500,]
    baseline_80=[0.765000, 0.770000, 0.720000, 0.770000, 0.737500,]
    new_method_80=[0.782500, 0.780000, 0.752500, 0.765000, 0.745000,]
    baseline_90=[0.712500, 0.740000, 0.700000, 0.737500, 0.727500,]
    new_method_90=[0.71300, 0.732500, 0.702500, 0.755000, 0.715000,]
    baseline_100=[0.710000, 0.742500, 0.692500, 0.737500, 0.710000,]
    new_method_100=[0.710000, 0.742500, 0.692500, 0.737500, 0.710000, ]
    pair_t_test([0.7825,
0.795,
0.78,
0.77,
0.765,], [0.785,
0.795,
0.775,
0.7875,
0.7825])

plot = False
if plot:
    plt.plot(scale_cv,glsvm_mean_02,label="Ours: G-LSVM ",color="red",linewidth=5)
    plt.errorbar(scale_cv, glsvm_mean_02, yerr=glsvm_std_02, fmt='o', color="red",linewidth=3) 
    
    plt.plot(scale_cv,lsvm_mean_00,label="Baseline: LSVM ",color="green",linewidth=5)
    plt.errorbar(scale_cv, lsvm_mean_00, yerr=lsvm_std_00, fmt='o', color="green",linewidth=3)
        
    axes=plt.gca()
    #         axes.set_ylim([.50, .65])
    axes.set_xlim([25,105])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.xlabel("Scale", fontsize=50)
    plt.ylabel("mAP", fontsize=50)
    #     plt.ylim(min(min(y_train), min(y_val),min(y_test)),max(max(y_train), max(y_val),max(y_test)))
    #     plt.title(res_typ+" of scale:%s, gain of test:%4.2f %%"%(scale,(y_test[y_val.index(max(y_val))]-y_test[0])*100))
    #     plt.title("object classification: test mAP")
    #     plt.axvline(x=scale_cv[y_val.index(max(y_val))], color= 'black', linestyle='dashed')
    plt.legend(loc='best',fancybox=True,framealpha=1, prop={'size':40})
    plt.show()      
