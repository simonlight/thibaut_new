package Symil_thibaut_raw;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * 
 */

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class Run {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

			File file = new File("/local/wangxin/Data/MilData/Musk/musk1norm.svm");
			List<TrainingSample<Bag>> listBag = MILReader.read(file);

			List<TrainingSample<LatentRepresentation<Bag,Integer>>> exampleTrain = new ArrayList<TrainingSample<LatentRepresentation<Bag,Integer>>>();
			for(int i=0; i<listBag.size(); i++) {
				exampleTrain.add(new TrainingSample<LatentRepresentation<Bag,Integer>>(new LatentRepresentation<Bag,Integer>(listBag.get(i).sample,0), listBag.get(i).label));
			}

			List<TrainingSample<LatentRepresentation<Bag,Integer>>> listPos = new ArrayList<TrainingSample<LatentRepresentation<Bag,Integer>>>();
			List<TrainingSample<LatentRepresentation<Bag,Integer>>> listNeg = new ArrayList<TrainingSample<LatentRepresentation<Bag,Integer>>>();
			for(TrainingSample<LatentRepresentation<Bag,Integer>> ts : exampleTrain) {
				//System.out.println(bag);
				if(ts.label == 1) {
					listPos.add(new TrainingSample<LatentRepresentation<Bag,Integer>>(new LatentRepresentation<Bag,Integer>(ts.sample.x,0), ts.label));
				}
				else if(ts.label == -1) {
					listNeg.add(new TrainingSample<LatentRepresentation<Bag,Integer>>(new LatentRepresentation<Bag,Integer>(ts.sample.x,0), ts.label));
				}
			}
			System.out.println("bag+ " + listPos.size() + " \t bag- " + listNeg.size());

			//setting nth fold
			int nPos = 3 * listPos.size() / 4;
			List<TrainingSample<LatentRepresentation<Bag,Integer>>> train = new ArrayList<TrainingSample<LatentRepresentation<Bag,Integer>>>();
			train.addAll(listPos.subList(0, nPos));
			int nNeg = 3 * listNeg.size() / 4;
			train.addAll(listNeg.subList(0, nNeg));
			List<TrainingSample<LatentRepresentation<Bag,Integer>>> test = new ArrayList<TrainingSample<LatentRepresentation<Bag,Integer>>>();
			test.addAll(listPos);
			test.addAll(listNeg);
			test.removeAll(train);
			
			System.out.println("Train=" + train.size() + "\ttest=" + test.size());

			SyMILGradientDescentBag lsvm = new SyMILGradientDescentBag();
			lsvm.setLambda(1e-4);
			lsvm.setMaxCCCPIter(15);
			lsvm.setMaxEpochs(100);
			lsvm.setVerbose(0);
			lsvm.setOptim(1);
			lsvm.setGamma(1);
			lsvm.setStochastic(true);
			lsvm.setInit(0);
			lsvm.train(train);
			
			System.out.println("Train");
			lsvm.accuracy(train);
			System.out.println("Test");
			lsvm.accuracy(test);
			
			System.out.println("Prediction test");
			for(int i=0; i<test.size(); i++) {
				TrainingSample<LatentRepresentation<Bag,Integer>> example =  test.get(i);
				double score = lsvm.valueOf(example.sample);
				int prediction = 0;
				if(score>0){	
					prediction = 1;
				}
				else {
					prediction = -1;
				}
				System.out.println("i=" + i + "\tprediction=" + prediction);
			}

		}
}
