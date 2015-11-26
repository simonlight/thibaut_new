package jstruct.data.uiuc.mac.ssvm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReaderXML;
import fr.durandt.jstruct.ssvm.multiclass.DoubleMulticlassSSVMFrankWolfeBlockCoordinate;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

public class MulticlassSSVMFrankWolfeBlockCoordinateTest {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/UIUCSports/";

	public static void main(String[] args) {

		int numWords = 4096;
		int maxIter = 150;

		double[] lambdaCV = {1e-9};
		double[] epsilonCV = {1e-2};
		int[] scaleCV = {100};
		//int[] splitCV = {1,2,3,4,5};
		int[] splitCV = {1};

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean compute = true;
		String features = "caffe_vgg19_6_relu_vd_multiscale_256_384_512";

		for(int scale : scaleCV) {
			for(int split : splitCV) {
				String cls = String.valueOf(split);

				String classifierDir = simDir + "classifier/ssvm/FrankWolfe_BlockCoordinate/" + features + "/";
				String inputDir = simDir + "Split_" + cls + "/files_BagImageRegion/" + features + "/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						DoubleMulticlassSSVMFrankWolfeBlockCoordinate svm = new DoubleMulticlassSSVMFrankWolfeBlockCoordinate(); 
						svm.setLambda(lambda);
						svm.setMaxIter(maxIter);

						String suffix = "_" + svm.toString();
						boolean testFile = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
						if(!testFile) {
							compute = true;
						}
					}
				}

				if(compute) {
					List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReaderXML.readBagImageRegion(new File(inputDir + "/multiclass_train_scale_" + scale + ".xml"), numWords, true, true, null, true, 0);
					List<STrainingSample<double[], Integer>> exampleTrain = new ArrayList<STrainingSample<double[], Integer>>();
					for(int i=0; i<listTrain.size(); i++) {
						exampleTrain.add(new STrainingSample<double[], Integer>(listTrain.get(i).input.getInstance(0), listTrain.get(i).output));
						/*double[] feature = new double[1000];
						for(int d=0; d<1000; d++) {
							feature[d] = exampleTrain.get(i).input[d];
						}
						exampleTrain.get(i).input = feature;*/
					}

					List<STrainingSample<BagImageRegion, Integer>> listTest = BagReaderXML.readBagImageRegion(new File(inputDir + "/multiclass_test_scale_" + scale + ".xml"), numWords, true, true, null, true, 0);
					List<STrainingSample<double[],Integer>> exampleTest = new ArrayList<STrainingSample<double[],Integer>>();
					for(int i=0; i<listTest.size(); i++) {
						exampleTest.add(new STrainingSample<double[], Integer>(listTest.get(i).input.getInstance(0), listTest.get(i).output));
						/*double[] feature = new double[1000];
						for(int d=0; d<1000; d++) {
							feature[d] = exampleTest.get(i).input[d];
						}
						exampleTest.get(i).input = feature;*/
					}

					for(double epsilon : epsilonCV) {
						for(double lambda : lambdaCV) {

							DoubleMulticlassSSVMFrankWolfeBlockCoordinate svm = new DoubleMulticlassSSVMFrankWolfeBlockCoordinate(); 
							svm.setLambda(lambda);
							svm.setMaxIter(maxIter);

							String suffix = "_" + svm.toString();
							boolean testFile = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
							if(compute || !testFile) {
								svm.train(exampleTrain);
								double acc = svm.multiclassAccuracy(exampleTrain);
								System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda + "\tepsilon= " + epsilon);

								acc = svm.multiclassAccuracy(exampleTest);
								System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda + "\tepsilon= " + epsilon);
								System.out.println("\n");
							}
						}
					}
				}
			}
		}
	}

	public static boolean testPresenceFile(String dir, String test) {
		boolean testPresence = false;

		File classifierDir = new File(dir);

		if(classifierDir.exists()) {
			String[] f = classifierDir.list();
			//System.out.println(Arrays.toString(f));

			for(String s : f) {
				if(s.contains(test)) {
					testPresence = true;
				}
			}
		}
		System.out.println("presence " + testPresence + "\t" + dir + "\t" + test);
		return testPresence;
	}
}
