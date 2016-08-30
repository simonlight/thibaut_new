import os
ori = "/local/wangxin/Data/ferrari_gaze/example_files"
des = "/local/wangxin/Data/ferrari_gaze/example_files_with_gaze_annotation"
if not os.path.exists(des):
    os.makedirs(des)

# categories = ["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
# categories = [
#             "apple-pie",
#             "bread-pudding",
#             "beef-carpaccio",
#             "beet-salad",
#             "chocolate-cake",
#             "chocolate-mousse",
#             "donuts",
#             "beignets",
#             "eggs-benedict",
#             "croque-madame",
#             "gnocchi",
#             "shrimp-and-grits",
#             "grilled-salmon",
#             "pork-chop",
#             "lasagna",
#             "ravioli",
#             "pancakes",
#             "french-toast",
#             "spaghetti-bolognese",
#             "pad-thai"        
#             ]
categories = [
              "dog", "cat", 
            "motorbike", "boat", 
            "aeroplane", "horse" ,
            "cow", "sofa", 
            "diningtable", "bicycle"
              ]
scale_cv = ['100','90','80','70','60', '50', '40', '30']

for c in categories:
    for s in scale_cv:
        name = '_'.join([c,'trainval', 'scale', s, 'matconvnet_m_2048_layer_20.txt'])
        train_name = '_'.join([c,'train', 'scale', s, 'matconvnet_m_2048_layer_20.txt'])
        valval_name = '_'.join([c,'valval', 'scale', s, 'matconvnet_m_2048_layer_20.txt'])
        valtest_name = '_'.join([c,'valtest', 'scale', s, 'matconvnet_m_2048_layer_20.txt'])
        
        f = open(ori+'/'+s+'/'+name)
        total_num=int(f.readline())
        train_num = int(total_num*0.8)
        valval_num = int(total_num*0.1)
        valtest_num = total_num - train_num - valval_num
        
        train_f = open(des+'/'+s+'/'+train_name,'w')
        train_f.write(str(train_num)+'\n')
        valval_f = open(des+'/'+s+'/'+valval_name,'w')
        valval_f.write(str(valval_num)+'\n')
        valtest_f = open(des+'/'+s+'/'+valtest_name,'w')
        valtest_f.write(str(valtest_num)+'\n')
        
        
        for cnt, line in enumerate(f):
            if cnt<train_num:
                train_f.write(line)
            elif cnt<train_num+valval_num:
                valval_f.write(line)
            else:
                valtest_f.write(line)
            