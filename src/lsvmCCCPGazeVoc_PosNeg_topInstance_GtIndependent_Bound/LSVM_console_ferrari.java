package lsvmCCCPGazeVoc_PosNeg_topInstance_GtIndependent_Bound;

public class LSVM_console_ferrari {
	public static void main(String[] args) {
	
		String dataSource= "local";//local or other things
		String gazeType = "ferrari";
		String taskName = "npglsvm_ferrari_traintrainlist_testtestlist_5split_weighted_gaze/";
		double[] lambdaCV = {1e-4};
	    double[] epsilonCV = {0};
//	    String[] classes = {args[0]};
	    int maxK=10;
//		int[] scaleCV = {Integer.valueOf(args[1])};
	//	String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
	//	String[] classes = {"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
	//	int[] scaleCV = {90,80,70,60,50,40,30};
		int[] scaleCV = {100};
		String[] classes = {"bicycle"};
	//    double[] tradeoffCV = {0, 0.5, 1};
	    double[] posTradeoffCV = {0, 0.1, 0.2, 0.5};
	    double[] negTradeoffCV = {0, 0.001, 0.01, 0.1,0.2};
	    
	    //Variables we may change
		int foldNum=10;
	    int minCCCPIter = 5;
		int maxCCCPIter = 1000;
		int maxSGDEpochs = 100;
		boolean stochastic = true;
		boolean saveClassifier = true;
	    boolean loadClassifier = false;
	    
	    //Variables we do not change
		int optim = 2;
		int randomSeed=1;
		int numWords = 2048;
		boolean hnorm = false;
		String gazeJmapFolder = "et_POET_gaze_ratio_jmap/";
		
		String sourceDir = new String();
		String resDir = new String();
	
		if (dataSource=="local"){
			sourceDir = "/local/wangxin/Data/ferrari_gaze/";
			resDir = "/local/wangxin/results/ferrari_gaze/glsvm_pos_neg/";
		}
		else if (dataSource=="big"){
			sourceDir = "/home/wangxin/Data/ferrari_gaze/";
			resDir = "/home/wangxin/results/ferrari_gaze/glsvm_pos_neg/";
		}
	
		LSVM_common_console.console(dataSource, gazeType, taskName, sourceDir, resDir, gazeJmapFolder, maxCCCPIter, minCCCPIter, maxSGDEpochs, optim, numWords, foldNum, randomSeed, stochastic, saveClassifier, loadClassifier, hnorm, lambdaCV, epsilonCV, posTradeoffCV, negTradeoffCV, classes, scaleCV, maxK);

	}
}
