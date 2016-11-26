cls_arr=("apple-pie" "bread-pudding" "beef-carpaccio" "beet-salad" "chocolate-cake" "chocolate-mousse" "donuts" "beignets" "eggs-benedict" "croque-madame" "gnocchi" "shrimp-and-grits" "grilled-salmon" "pork-chop" "lasagna" "ravioli" "pancakes" "french-toast" "spaghetti-bolognese" "pad-thai")
#cls_arr=("beignets")
#scale_arr=("100" "90" "80" "70" "60" "50" "40" "30")
scale_arr=("50" "40" "30")
k='oarsub -p "host='"'"'big30'"'"'  or host='"'"'big30'"'"' or host='"'"'big31'"'"' or host='"'"'big20'"'"' or host='"'"'big20'"'"' " -l "nodes=1/core=3,walltime=1000:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -Xms4g -Xmx10g -XX:-UseGCOverheadLimit -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmCCCPGazeVoc_PosNeg_topInstance/LSVM_console_ufood'
end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
    done
done
