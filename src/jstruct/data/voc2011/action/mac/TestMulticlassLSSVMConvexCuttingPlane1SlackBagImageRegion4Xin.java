/**
 * 
 */
package jstruct.data.voc2011.action.mac;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.multiclass.FastMulticlassLSSVMCuttingPlane1SlackBagImage;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImage;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class TestMulticlassLSSVMConvexCuttingPlane1SlackBagImageRegion4Xin {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr_2013_tutoriel/";

	private static int numWords = 2048;

	public static void main(String[] args) {


		List<STrainingSample<BagImage, Integer>> listTrain = BagReader.readBagImage("/local/wangxin/Data/ferrari_gaze/example_files/90/horse_train_scale_90_matconvnet_m_2048_layer_20.txt", numWords, true, true, null, true, 0);
		List<STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>>();
		for(int i=0; i<listTrain.size(); i++) {
			exampleTrain.add(new STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>(new LatentRepresentation<BagImage, Integer>(listTrain.get(i).input,0), listTrain.get(i).output));
		}

		List<STrainingSample<BagImage, Integer>> listTest = BagReader.readBagImage("/local/wangxin/Data/ferrari_gaze/example_files/90/horse_val_scale_90_matconvnet_m_2048_layer_20.txt", numWords, true, true, null, true, 0);
		List<STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>>();
		for(int i=0; i<listTest.size(); i++) {
			exampleTest.add(new STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>(new LatentRepresentation<BagImage, Integer>(listTest.get(i).input,0), listTest.get(i).output));
		}

		FastMulticlassLSSVMCuttingPlane1SlackBagImage classifier = new FastMulticlassLSSVMCuttingPlane1SlackBagImage(); 
		classifier.setLambda(1e-4);
		classifier.setEpsilon(1e-3);
		classifier.setCpmax(500);
		classifier.setCpmin(2);
		classifier.setVerbose(1);

		classifier.train(exampleTrain);
		//double ap = classifier.averagePrecision(exampleTrain);
		//System.err.println("train - ap= " + ap);

//		ap = classifier.averagePrecision(exampleTest);
//		System.err.println("test - ap= " + ap );
		System.out.println("\n");

		//System.out.println(Arrays.toString())

		for(STrainingSample<LatentRepresentation<BagImage, Integer>,Integer> ex : exampleTrain) {
			Object[] res = classifier.predictionOutputLatent(ex.input.x);
			Integer h = (Integer)res[1];
			System.out.println(h);
		}
	}

}
