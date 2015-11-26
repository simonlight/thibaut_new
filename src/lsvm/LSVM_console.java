/**
 * 
 */
package lsvm;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.Bag;
import fr.durandt.jstruct.variable.BagImage;
import fr.durandt.jstruct.variable.BagLabel;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LSVM_console {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr_2013_tutoriel/";

	private static int numWords = 2048;

	public static void main(String[] args) {
		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTrain = BagReader.readBagImageLatent("/local/wangxin/Data/ferrari_gaze/example_files/50/horse_train_scale_50_matconvnet_m_2048_layer_20.txttest", numWords, true, true, null, true, 0);
//		List<TrainingSample<BagImage, Integer>> listTrain = BagReader.readBagImage("/local/wangxin/Data/ferrari_gaze/example_files/90/horse_train_scale_90_matconvnet_m_2048_layer_20.txt", numWords, true, true, null, true, 0);
		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTest = BagReader.readBagImageLatent("/local/wangxin/Data/ferrari_gaze/example_files/50/horse_train_scale_50_matconvnet_m_2048_layer_20.txttest", numWords, true, true, null, true, 0);
		
		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTrain = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
		for(int i=0; i<listTrain.size(); i++) {
			exampleTrain.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTrain.get(i).sample.x,0), listTrain.get(i).label));
		}

		
		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTest = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
		for(int i=0; i<listTrain.size(); i++) {
			exampleTest.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTest.get(i).sample.x,0), listTest.get(i).label));
		}

//		train(List<TrainingSample<LatentRepresentation<X,H>>> l)
//		public void train(TrainingSample<LatentRepresentation<X,H>> t) {
		LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
		
		classifier.setLambda(1e-4);
		classifier.setVerbose(0);

		classifier.train(exampleTrain);
//		double ap = classifier.averagePrecision(exampleTrain);
		//System.err.println("train - ap= " + ap);

//		ap = classifier.averagePrecision(exampleTest);
//		System.err.println("test - ap= " + ap );
		System.out.println("\n");

		//System.out.println(Arrays.toString())

//		for(STrainingSample<LatentRepresentation<BagImage, Integer>,Integer> ex : exampleTrain) {
//			Object[] res = classifier.predictionOutputLatent(ex.input.x);
//			Integer h = (Integer)res[1];
//			System.out.println(h);
//		}
	}

}
