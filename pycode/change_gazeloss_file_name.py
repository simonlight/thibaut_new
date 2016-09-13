import os
ori = "/local/wangxin/Data/ferrari_gaze/ETLoss_ratio"
des = "/local/wangxin/Data/ferrari_gaze/gazeloss_folder_noclasswise"

categories = [
            "dog", "cat", 
            "motorbike", "boat", 
            "aeroplane", "horse" ,
            "cow", "sofa", 
            "diningtable", "bicycle"
              ]
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
ori_cv = ['1','4']
scale_cv = ['100','90']

for c in categories:
    for old_s in ori_cv:
        print c,old_s
        folder = ori + '/' + c + '/' +old_s
        for filename in os.listdir(folder):
            f= open(folder + '/' + filename)
            v=float(f.readline().strip())
            new_v = 1-v
            new_folder = des  +'/' +scale_cv[ori_cv.index(old_s)] + '/'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            new_f = open(new_folder+filename, 'w')
            new_f.write(str(new_v)+'\n')
            new_f.close()     
            
            
for c in categories:
    for s in scale_cv:
        print (c,s)
        folder = des  + '/' +s+'/'
        new_folder = new_des +'/' +s+'/'
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        for filename in os.listdir(folder):
            new_filename = '_'.join(filename.split('_')[1:])
            os.system('cp %s %s'%(folder+filename, new_folder+new_filename))       