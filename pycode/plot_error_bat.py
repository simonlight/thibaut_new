import collections
import matplotlib.pyplot as plt
import numpy as np
import os
plt.figure(figsize=(8,4))
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)  
scale_cv = ['100','90','80','70','60', '50', '40', '30']


lsvm_mean=[88.2,89.24,89.63,89.5,89.2,88.8,87.2,84.18]
lsvm_std=[1.2,1.15,1.2,1.11,0.99,.87,1.13,0.52]
glsvm_mean=[88.2,89.27,89.68,89.62,89.53,88.95,87.73,85.13]
glsvm_std=[1.2,1.17,1.2,1.1,.87,1.02,1.13,.49]
#obj
lsvm_mean=[60.8, 62.6,63.8,64.9,64.3,63.8,62.4,58.4]
lsvm_std=[1.2, 1.0,1.2,1.3,1.2,0.4,1.0,1.6]
glsvm_mean=[60.8, 62.8,64.0,65.4,65.5,65.6,65.24,62.16]
glsvm_std=[1.2, 0.8,1.0,1.1,1.1,.75,1.16,1.06]

#act


plt.plot(scale_cv,glsvm_mean,label="G-LSVM ",color="red",linewidth=5)
plt.errorbar(scale_cv, glsvm_mean, yerr=glsvm_std, fmt='o', color="red",linewidth=3) 

plt.plot(scale_cv,lsvm_mean,label="LSVM ",color="green",linewidth=5)
plt.errorbar(scale_cv, lsvm_mean, yerr=lsvm_std, fmt='o', color="green",linewidth=3)
    
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
