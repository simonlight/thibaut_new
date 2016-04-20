/**
 * 
 */
package Symil_Musk;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import Symil_Musk.LSVMGradientDescentBag;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.LatentRepresentationSymil;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class LSVM_console_Musk {
	public static void main(String[] args) {
	
		double[] lambdaCV = {1e-4};
	    double[] epsilonCV = {0};
	    int foldNum=10;
	    int maxCCCPIter = 1000;
	    int minCCCPIter = 10;
	    double[] nbdCV = {1.0};

		int maxSGDEpochs = 100;
		
		boolean stochastic = true;
	    
		int optim = 2;
		int numWords = 166;
	    
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
	
	String listTrainPath =  "/local/wangxin/Data/MilData/example_files/musk1/example_file.txt";

	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> fullList = BagReader.readBagImageLatent(listTrainPath, numWords, true, true, null, true, 0, "local");

	for(double epsilon : epsilonCV) {
    	for(double lambda : lambdaCV) {
    		for (double nbd : nbdCV){
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
		    			int train_neg_cnt=0;

//		    			for (int c_trainList=0; c_trainList<trainList.size();c_trainList++){
//		    				if (trainList.get(c_trainList).label==1){
//		    					train_pos_cnt+=1;
//		    				}
//		    				else{
//		    					train_neg_cnt+=1;
//		    				}
//		    			}
//		    			if (train_pos_cnt<20){
//		    				System.exit(0);
//		    			}
		    			
		    			List<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>> exampleTrain = new ArrayList<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>>();
		    			List<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>> exampleTest = new ArrayList<TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>>();

//		    			for(int j=0; j<testList.size(); j++) {
//	    					if (testList.get(j).label==1 &&train_pos_cnt<35) {
//		    					train_pos_cnt+=1;
//			    				exampleTrain.add(new TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>(new LatentRepresentationSymil<BagImage, Integer,Integer>(testList.get(j).sample.x,0,0), testList.get(j).label));
//		    					continue;
//	    					}
//	    					if(testList.get(j).label==-1&&train_neg_cnt<33){
//		    					train_neg_cnt+=1;
//	    						exampleTrain.add(new TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>(new LatentRepresentationSymil<BagImage, Integer,Integer>(testList.get(j).sample.x,0,0), testList.get(j).label));
//	    						continue;	
//	    					}
//    						exampleTest.add(new TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>(new LatentRepresentationSymil<BagImage, Integer,Integer>(testList.get(j).sample.x,0,0), testList.get(j).label));
//		    			}
//		    			System.out.println("+:"+train_pos_cnt);
//		    			System.out.println("-:"+train_neg_cnt);
		    			
		    			for(int j=0; j<trainList.size(); j++) {
		    				exampleTrain.add(new TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>(new LatentRepresentationSymil<BagImage, Integer,Integer>(trainList.get(j).sample.x,0,0), trainList.get(j).label));
		    			}
		    			for(int j=0; j<testList.size(); j++) {
							exampleTest.add(new TrainingSample<LatentRepresentationSymil<BagImage, Integer,Integer>>(new LatentRepresentationSymil<BagImage, Integer,Integer>(testList.get(j).sample.x,0,0), testList.get(j).label));
						}

						LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
					
							classifier.setOptim(optim);
							classifier.setMaxCCCPIter(maxCCCPIter);
							classifier.setMinCCCPIter(minCCCPIter);
							classifier.setEpsilon(epsilon);
							classifier.setLambda(lambda);
							classifier.setStochastic(stochastic);
							classifier.setVerbose(0);
							classifier.setNbd(nbd);
							classifier.setMaxEpoch(maxSGDEpochs);
							classifier.train(exampleTrain);
							
							classifier.optimizeLatent(exampleTest);
							cv_results += classifier.accuracy(exampleTest);
	    			}
	    		System.err.println(cv_results);
	    		final_results +=cv_results/foldNum;	
    		}
    		System.out.format("after %d %d-fold test, final cv=%f",exp_num,foldNum,final_results/exp_num);
    	}}}}

}
