cls_arr=("jumping" "phoning" "playinginstrument" "reading" "ridingbike" "ridinghorse" "running" "takingphoto" "usingcomputer" "walking")
#cls_arr=("takingphoto" "usingcomputer" "walking")
#scale_arr=("90" "80" "70" "60" "50" "40" "30")
#cls_arr=("phoning")

scale_arr=("30")
#k='oarsub -p "host='"'"'big17'"'"' or host='"'"'big18'"'"' or host='"'"'big19'"'"' or host='"'"'big20'"'"' " -l "nodes=1/core=1,walltime=500:0:0" --notify "mail:xin.wang@lip6.fr" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/lssvm_wang/src:. 
#data/uiuc/mac/StefanLSSVMMulticlassTestET'

k='oarsub -p "host='"'"'big18'"'"' or host='"'"'big20'"'"'" -l "nodes=1/core=8,walltime=500:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmStandardVoc/LSVM_console_stefan'

#k='oarsub -p "host='"'"'big$1'"'"' " -l "nodes=1/core=1,walltime=500:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/lssvm_wang/src:. 
#data/uiuc/mac/LSVM_console_stefan'


end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
    done
done
