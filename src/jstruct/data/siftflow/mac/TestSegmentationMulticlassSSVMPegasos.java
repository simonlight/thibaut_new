package jstruct.data.siftflow.mac;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.ssvm.segmentation.SegmentationAD3SSVMPegasos;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageSeg;

public class TestSegmentationMulticlassSSVMPegasos {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/SiftFlowDataset/";

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[] lambdaCV = {1e-4};
		Integer[] maxIterCV = {10};

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("maxIter " + Arrays.toString(maxIterCV) + "\n");

		boolean recompute = false;
		String features = "places";
		int verbose = 1;
		int numWords = 4096;

		String classifierDir = simDir + "classifier/ssvm/pegasos/AD3/" + features + "_caffe_6_relu/";
		String inputDir = simDir + "/files/";

		System.out.println("classifierDir: " + classifierDir + "\n");

		String[] clsCV = {"0.500000"};

		for(String cls : clsCV) {

			boolean compute = false;
			for(int maxIter : maxIterCV) {
				for(double lambda : lambdaCV) {

					SegmentationAD3SSVMPegasos classifier = new SegmentationAD3SSVMPegasos();
					classifier.setLambda(lambda);
					classifier.setVerbose(verbose);
					classifier.setMaxIterations(maxIter);

					String suffix = "_" + classifier.toString();
					File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + suffix);
					if(fileClassifier == null) {
						compute = true;
					}
				}
			}

			if(compute) {
				// Read train set
				List<STrainingSample<BagImageSeg,Integer[]>> exampleTrain = BagReader.readBagImageSeg(new File(inputDir + "/segmentation_" + features + "_" + cls + "_train.txt"), numWords, true, true, null, true, 0);
				// Read test set
				List<STrainingSample<BagImageSeg,Integer[]>> exampleTest = BagReader.readBagImageSeg(new File(inputDir + "/segmentation_" + features + "_" + cls + "_test.txt"), numWords, true, true, null, true, 0);

				for(int maxIter : maxIterCV) {
					for(double lambda : lambdaCV) {

						SegmentationAD3SSVMPegasos classifier = new SegmentationAD3SSVMPegasos();
						classifier.setLambda(lambda);
						classifier.setVerbose(verbose);
						classifier.setMaxIterations(maxIter);

						String suffix = "_" + classifier.toString();
						File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + suffix);
						if(recompute || fileClassifier == null) {
							classifier.train(exampleTrain);
							double acc = classifier.evaluation(exampleTrain);
							System.out.println("TRAIN - SuperPixels loss= " + acc);
							//classifier.evaluationPixelAccuracy(listTrain);

							acc = classifier.evaluation(exampleTest);
							System.out.println("TEST - SuperPixels loss= " + acc);
							classifier.evaluationPixelAccuracy(exampleTest);
							acc = classifier.evaluationPerClass(exampleTest);

							System.err.println("TEST - " + cls + "\tacc= " + acc + "\tlambda= " + lambda + "\tmaxIter= " + maxIter);

							fileClassifier = new File(classifierDir + "/" + cls + "/" + cls + suffix + "_acc_" + acc + ".ser");
							fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
							System.out.println("save classifier " + fileClassifier.getAbsolutePath());
							// save classifier
							ObjectOutputStream oos = null;
							try {
								oos = new ObjectOutputStream(new FileOutputStream(fileClassifier.getAbsolutePath()));
								oos.writeObject(classifier);
							} 
							catch (FileNotFoundException e) {
								e.printStackTrace();
							} 
							catch (IOException e) {
								e.printStackTrace();
							}
							finally {
								try {
									if(oos != null) {
										oos.flush();
										oos.close();
									}
								} catch (IOException e) {
									e.printStackTrace();
								}
							}

							System.err.println("test - " + cls + "\tacc= " + acc + "\tlambda= " + lambda + "\tmaxIter= " + maxIter);
							System.out.println("\n");
						}
					}
				}
			}
		}
	}

	public static File testPresenceFile(String dir, String test) {
		boolean testPresence = false;
		File classifierDir = new File(dir);
		File file = null;
		if(classifierDir.exists()) {
			String[] f = classifierDir.list();
			//System.out.println(Arrays.toString(f));

			for(String s : f) {
				if(s.contains(test)) {
					testPresence = true;
					file = new File(dir + "/" + s);
				}
			}
		}
		System.out.println("presence " + testPresence + "\t" + dir + "\t" + test + "\tfile " + (file == null ? null : file.getAbsolutePath()));
		return file;
	}

}
