import os
ori = "/local/wangxin/Data/UPMC_Food_Gaze_20/vgg-m-2048_features"
des = "/local/wangxin/Data/UPMC_Food_Gaze_20/vgg-m-2048_features_new"

# categories = ["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
categories = [
            "apple-pie",
            "bread-pudding",
            "beef-carpaccio",
            "beet-salad",
            "chocolate-cake",
            "chocolate-mousse",
            "donuts",
            "beignets",
            "eggs-benedict",
            "croque-madame",
            "gnocchi",
            "shrimp-and-grits",
            "grilled-salmon",
            "pork-chop",
            "lasagna",
            "ravioli",
            "pancakes",
            "french-toast",
            "spaghetti-bolognese",
            "pad-thai"        
            ]
scale_cv = ['100','90','80','70','60', '50', '40', '30']

for c in categories:
    for old_s in scale_cv:
        old_folder = ori + '/' + c + '/' +old_s
        new_folder = des + '/' + old_s + '/' +c
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        for filename in os.listdir(old_folder):
            os.system("cp %s %s"%(old_folder + '/' + filename, new_folder+ '/' + filename))
                      
