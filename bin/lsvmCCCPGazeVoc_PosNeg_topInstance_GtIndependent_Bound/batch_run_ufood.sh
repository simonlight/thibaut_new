#cls_arr=("apple-pie" "bread-pudding" "beef-carpaccio" "beet-salad" "chocolate-cake" "chocolate-mousse" "donuts" "beignets" "eggs-benedict" "croque-madame" "gnocchi" "shrimp-and-grits" "grilled-salmon" "pork-chop" "lasagna" "ravioli" "pancakes" "french-toast" "spaghetti-bolognese" "pad-thai")
cls_arr=("beignets")
#scale_arr=("100" "90" "80" "70" "60" "50" "40" "30")
#scale_arr=("100" "90" "80" "70" "60" "50")
scale_arr=("90")
k='oarsub -p "host='"'"'big16'"'"'  or host='"'"'big16'"'"'  " -l "nodes=1/core=1,walltime=1000:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -Xms2g -Xmx8000m -XX:-UseGCOverheadLimit -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmCCCPGazeVoc_PosNeg_topInstance_GtIndependent_Bound/LSVM_console_ufood'
end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
        sleep 3
    done
done
