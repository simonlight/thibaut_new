#cls_arr=("aeroplane" "cow" "dog" "cat" "motorbike" "boat"  "horse"  "sofa" "diningtable" "bicycle")
cls_arr=("aeroplane" "bicycle")
#scale_arr=("100" "90" "80" "70" "60" "50" "40" "30")
scale_arr=("30" )

k='oarsub -p "  host='"'"'big19'"'"'  or host='"'"'big19'"'"' or host='"'"'big19'"'"'  " -l "nodes=1/core=6,walltime=1000:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -Xms40000m -Xmx40000m -XX:-UseGCOverheadLimit -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmCCCPGazeVoc_PosNeg_topInstance/LSVM_console_ferrari'


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
