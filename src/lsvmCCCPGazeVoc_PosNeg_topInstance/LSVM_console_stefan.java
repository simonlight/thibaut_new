package lsvmCCCPGazeVoc_PosNeg_topInstance;

public class LSVM_console_stefan {
	public static void main(String[] args) {
	
		String dataSource= "local";//local or other things
		String gazeType = "stefan";
		String taskName = "test_positivegroundtruth_npglsvm_stefan_traintrainlist_testtestlist_5split/";
		double[] lambdaCV = {1e-4};
	    double[] epsilonCV = {0};
	    int maxK=10;

	//	String[] classes = {args[0]};
	//	int[] scaleCV = {Integer.valueOf(args[1])};
	//	String[] classes = {"jumping", "phoning", "playinginstrument", "reading" ,"ridingbike", "ridinghorse" ,"running" ,"takingphoto" ,"usingcomputer", "walking"};
	    String[] classes = {"jumping"};
	//    int[] scaleCV = {90,80,70,60,50,40,30};
	    int[] scaleCV = {100};
	    
	    double[] posTradeoffCV = {0.2};
	    double[] negTradeoffCV = {0, 0.001, 0.01, 0.1 ,0.2,0.5, 1.0};
	    
	    //Variables we may change
		int foldNum=5;
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
	    String gazeJmapFolder = "et_voc2012_action_gaze_ratio_jmap/";

		String sourceDir = new String();
		String resDir = new String();
		
		if (dataSource=="local"){
			sourceDir = "/local/wangxin/Data/full_stefan_gaze/";
			resDir = "/local/wangxin/results/full_stefan_gaze/glsvm_pos_neg/";
		}
		else if (dataSource=="big"){
			sourceDir = "/home/wangxin/Data/full_stefan_gaze/";
			resDir = "/home/wangxin/results/full_stefan_gaze/glsvm_pos_neg/";
		}
		
		LSVM_common_console.console(dataSource, gazeType, taskName, sourceDir, resDir, gazeJmapFolder, maxCCCPIter, minCCCPIter, maxSGDEpochs, optim, numWords, foldNum, randomSeed, stochastic, saveClassifier, loadClassifier, hnorm, lambdaCV, epsilonCV, posTradeoffCV, negTradeoffCV, classes, scaleCV, maxK);
	}
}
