import collections
content = collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(lambda:None))))
f = open("/home/wangxin/kepler/results/full_stefan_gaze/glsvm_pos_neg/lsvm_cccpgaze_posneg_inverse_noreweighting/ap_summary.txt")

for line in f:
    category, scale, tradeoff, index, ap_test,ap_train = line.strip().split(' ')
    category = category.split(':')[1]
    scale = scale.split(':')[1]
    gamma_pos, gamma_neg = tradeoff.split(':')[1].split('_')
    index = index.split(':')[1]
    ap_test = ap_test.split(':')[1][:6]
    ap_train = ap_train.split(':')[1][:5]
    
    content[scale][category][gamma_neg][gamma_pos] = float(ap_test)

action_names = ["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
scale_cv = ['90','80','70','60', '50', '40', '30']

for scale in scale_cv:
    print scale
    sum1=[]
    sum2=[]
    sum3=[]
    for action in action_names:
        max_pos_neg = -1
        max_pos_neg_index = [0,0]
        for gamma_neg in content[scale][action].keys():
            if max(content[scale][action][gamma_neg].values()) > max_pos_neg:
                max_pos_neg = max(content[scale][action][gamma_neg].values())
        sum1.append(100*max(content[scale][action]['0.0'].values() ) )
        sum2.append(100* max_pos_neg)
        sum3.append(100*content[scale][action]['0.0']["0.0"])

        print action, "\t &", 100*max(content[scale][action]['0.0'].values() ) , "&",100* max_pos_neg, "&",content[scale][action]['0.0']["0.0"], "\\\ \hline"
    print "map", "\t &", sum(sum1)/10, "&", sum(sum2)/10,"&", sum(sum3)/10, "\\\ \hline"
        
    
