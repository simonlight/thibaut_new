/**
 * 
 */
package lsvmCCCPMusk;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LSVM_console_Musk {
	public static void main(String[] args) {
	
	double[] lambdaCV = {0.001};
    double[] epsilonCV = {0};
    int foldNum=10;
	int maxCCCPIter = 1000;
	int minCCCPIter = 1;

	int maxSGDEpochs = 100;
	
	boolean semiConvexity = true;
	boolean stochastic = false;
    
	int optim = 2;
	int numWords = 230;
    
	int exp_num=10;
	
	System.out.println("experiment detail: "
			+ "\nlambda CV:\t" + Arrays.toString(lambdaCV)
			+ "\nepsilon CV:\t" + Arrays.toString(epsilonCV)
			+ "\noptim:\t"+optim
			+ "\nmaxCCCPIter:\t"+maxCCCPIter
			+ "\nminCCCPIter:\t"+minCCCPIter
			+ "\nmaxSGDEpochs:\t"+maxSGDEpochs
			+ "\nnumWords:\t"+numWords
		    );
	
			String listTrainPath =  "/local/wangxin/Data/MilData/example_files/tiger/example_file.txt";

    		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> fullList = BagReader.readBagImageLatent(listTrainPath, numWords, true, true, null, true, 0, "local");

	    	for(double epsilon : epsilonCV) {
		    	for(double lambda : lambdaCV) {
		    		double final_results=0;
		    		for (int k=0;k<exp_num;k++){
			    		double cv_results=0;
						Collections.shuffle(fullList);
						int listsize = fullList.size();
			    			for (int i=0;i<foldNum; i++){
				    			int fromIndex = listsize * i/foldNum;
				    			int toIndex = listsize * (i+1)/foldNum;
				    			List<TrainingSample<LatentRepresentation<BagImage,Integer>>> testList = fullList.subList(fromIndex, toIndex);
				    			List<TrainingSample<LatentRepresentation<BagImage,Integer>>> trainList_1 = fullList.subList(0, fromIndex);
				    			List<TrainingSample<LatentRepresentation<BagImage,Integer>>> trainList_2 = fullList.subList(toIndex, listsize);
				    			List<TrainingSample<LatentRepresentation<BagImage,Integer>>> trainList = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>> ();
				    			trainList.addAll(trainList_1);
				    			trainList.addAll(trainList_2);
				    			int train_pos_cnt=0;
				    			for (int c_trainList=0; c_trainList<trainList.size();c_trainList++){
				    				if (trainList.get(c_trainList).label==1){
				    					train_pos_cnt+=1;
				    				}
				    			}
				    			System.out.println("number of pos_train:"+train_pos_cnt);
				    			if (train_pos_cnt<20){
				    				System.exit(0);
				    			}
				    			
				    			List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTrain = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
				    			List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTest = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
		
				    			for(int j=0; j<trainList.size(); j++) {
									exampleTrain.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(trainList.get(j).sample.x,0), trainList.get(j).label));
								}
				    			for(int j=0; j<testList.size(); j++) {
									exampleTest.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(testList.get(j).sample.x,0), testList.get(j).label));
								}
		
								LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
							
									classifier.setOptim(optim);
									classifier.setMaxCCCPIter(maxCCCPIter);
									classifier.setMinCCCPIter(minCCCPIter);
									classifier.setSemiConvexity(semiConvexity);
									classifier.setEpsilon(epsilon);
									classifier.setLambda(lambda);
									classifier.setStochastic(stochastic);
									classifier.setVerbose(0);
		
									classifier.train(exampleTrain);
									cv_results += classifier.accuracy(exampleTest);
			    			}
			    		System.err.println(cv_results);
			    		final_results +=cv_results/foldNum;	
		    		}
		    		System.out.format("after %d %d-fold test, final cv=%f",exp_num,foldNum,final_results/exp_num);
		    	}}}

}
