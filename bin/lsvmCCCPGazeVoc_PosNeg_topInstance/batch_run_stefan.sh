cls_arr=("jumping" "phoning" "playinginstrument" "reading" "ridingbike" "ridinghorse" "running" "takingphoto" "usingcomputer" "walking")
scale_arr=("100" "90" "80" "70" "60" "50" "40" "30")
#cls_arr=("jumping")
#scale_arr=("90")

k='oarsub -p " host='"'"'big7'"'"'  or host='"'"'big9'"'"' or host='"'"'big18'"'"' or host='"'"'big19'"'"' or host='"'"'big31'"'"'  " -l "nodes=1/core=1,walltime=1000:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -Xms5000m -Xmx12000m -XX:-UseGCOverheadLimit -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmCCCPGazeVoc_PosNeg_topInstance/LSVM_console_stefan'

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
