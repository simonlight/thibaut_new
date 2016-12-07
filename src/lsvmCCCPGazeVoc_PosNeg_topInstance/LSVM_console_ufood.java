package lsvmCCCPGazeVoc_PosNeg_topInstance;

public class LSVM_console_ufood {
	public static void main(String[] args) {
	
		String dataSource= "big";//local or other things
		String gazeType = "ufood";
		String taskName = "complement_pad_thai_topk_ICIP_loss_weighted_food_10split_full/";
		double[] lambdaCV = {1e-4};
	    double[] epsilonCV = {0};
//	    int[] scaleCV = {100};
	    int maxK=10;

		String[] classes = {args[0]};
		int[] scaleCV = {Integer.valueOf(args[1])};
//	    int[] scaleCV = {50};
//	    String[] classes={"apple-pie"};
//	    String[] classes={
//				"apple-pie",
//				"bread-pudding",
//				"beef-carpaccio",
//				"beet-salad",
//				"chocolate-cake",
//				"chocolate-mousse",
//				"donuts",
//				"beignets",
//				"eggs-benedict",
//				"croque-madame",
//				"gnocchi",
//				"shrimp-and-grits",
//				"grilled-salmon",
//				"pork-chop",
//				"lasagna",
//				"ravioli",
//				"pancakes",
//				"french-toast",
//				"spaghetti-bolognese",
//				"pad-thai"		
//				};
	    
	    double[] posTradeoffCV = {0.0,0.1,0.2};
	    double[] negTradeoffCV = {0.001};
	    //Variables we may change
	    int minCCCPIter = 5;
		int maxCCCPIter = 100;
		int maxSGDEpochs = 100;
		boolean stochastic = true;
		boolean saveClassifier = true;
		boolean loadClassifier = false;

		//Variables we do not change
		int optim = 2;
		int foldNum=10;
		int randomSeed=1;
		int numWords = 2048;
		boolean hnorm = false;
		String gazeJmapFolder = "et_upmc_gaze_ratio_jmap/";
		
		String sourceDir = new String();
		String resDir = new String();
	
		if (dataSource=="local"){
			sourceDir = "/local/wangxin/Data/UPMC_Food_Gaze_20/";
			resDir = "/local/wangxin/results/upmc_food/";
		}
		else if (dataSource=="big"){
			sourceDir = "/home/wangxin/Data/UPMC_Food_Gaze_20/";
			resDir = "/home/wangxin/results/upmc_food/";
		}
		
		LSVM_common_console.console(dataSource, gazeType, taskName, sourceDir, resDir, gazeJmapFolder, maxCCCPIter, minCCCPIter, maxSGDEpochs, optim, numWords, foldNum, randomSeed, stochastic, saveClassifier, loadClassifier, hnorm, lambdaCV, epsilonCV, posTradeoffCV, negTradeoffCV, classes, scaleCV, maxK);
		
	}
	

}
