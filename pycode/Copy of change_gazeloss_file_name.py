import os
des = "/local/wangxin/Data/ferrari_gaze/gazeloss_folder_noclasswise_reduced"
new_des = "/local/wangxin/Data/ferrari_gaze/gazeloss_folder_noclasswise_reduced_new"
categories = [
            "dog", 
#             "cat", 
#             "motorbike", "boat", 
#             "aeroplane", "horse" ,
#             "cow", "sofa", 
#             "diningtable", "bicycle"
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
scale_cv = [
#             '100',
#             '90',
#             '80',
#             '70',
#             '60',
#             '50',
#             '40',
                '30'
               ]

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