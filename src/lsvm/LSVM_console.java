/**
 * 
 */
package lsvm;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.Bag;
import fr.durandt.jstruct.variable.BagImage;
import fr.durandt.jstruct.variable.BagLabel;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.durandt.jstruct.util.AveragePrecision;;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LSVM_console {
	
	
	
	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr_2013_tutoriel/";

	private static int numWords = 2048;
	private static boolean saveClassifier = true;
	
	
	
	public static void main(String[] args) {
		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTrain = BagReader.readBagImageLatent("/local/wangxin/Data/ferrari_gaze/example_files/60/horse_train_scale_60_matconvnet_m_2048_layer_20.txt", numWords, true, true, null, true, 0);
//		List<TrainingSample<BagImage, Integer>> listTrain = BagReader.readBagImage("/local/wangxin/Data/ferrari_gaze/example_files/90/horse_train_scale_90_matconvnet_m_2048_layer_20.txt", numWords, true, true, null, true, 0);
		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTest = BagReader.readBagImageLatent("/local/wangxin/Data/ferrari_gaze/example_files/60/horse_valval_scale_60_matconvnet_m_2048_layer_20.txt", numWords, true, true, null, true, 0);
		
		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTrain = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
		for(int i=0; i<listTrain.size(); i++) {
			exampleTrain.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTrain.get(i).sample.x,0), listTrain.get(i).label));
		}

		
		List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTest = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
		for(int i=0; i<listTest.size(); i++) {
			exampleTest.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTest.get(i).sample.x,0), listTest.get(i).label));
		}

//		train(List<TrainingSample<LatentRepresentation<X,H>>> l)
//		public void train(TrainingSample<LatentRepresentation<X,H>> t) {
		LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
		
		classifier.setLambda(1e-4);
		classifier.setVerbose(0);

		classifier.train(exampleTrain);
		
//		if (saveClassifier){
//			// save classifier
//			
//			ObjectOutputStream oos = null;
//			try {
//				oos = new ObjectOutputStream(new FileOutputStream(fileClassifier.getAbsolutePath()));
//				oos.writeObject(classifier);
//			} 
//			catch (FileNotFoundException e) {
//				e.printStackTrace();
//			} 
//			catch (IOException e) {
//				e.printStackTrace();
//			}
//			finally {
//				try {
//					if(oos != null) {
//						oos.flush();
//						oos.close();
//					}
//				} catch (IOException e) {
//					e.printStackTrace();
//				}
//			}
//			System.out.println("wrote classifier successfully!");
//		}
		
		double ap_train = classifier.testAP(exampleTrain);
		System.err.println("train - ap= " + ap_train);

		double ap_test = classifier.testAP(exampleTest);
		System.err.println("test - ap= " + ap_test );
		System.out.println("\n");

		//System.out.println(Arrays.toString())

//		for(STrainingSample<LatentRepresentation<BagImage, Integer>,Integer> ex : exampleTrain) {
//			Object[] res = classifier.predictionOutputLatent(ex.input.x);
//			Integer h = (Integer)res[1];
//			System.out.println(h);
//		}
	}

}
