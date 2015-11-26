package jstruct.data.uiuc.mac.ssvm;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.ssvm.multiclass.ML3CuttingPlane1Slack;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Tests of ML3 on UIUC Sports with deep features
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class ML3CuttingPlane1SlackTest {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/UIUCSports/";

	public static void main(String[] args) {

		int numWords = 4096;
		int cpmax = 500;
		int cpmin = 5;

		double[] lambdaCV = {1e-5};
		double[] epsilonCV = {1e-3};
		double[] pCV = {1.5};
		Integer[] mCV = {5};
		//int[] splitCV = {1,2,3,4,5};
		int[] splitCV = {1};
		int scale = 100;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("p " + Arrays.toString(pCV));
		System.out.println("m " + Arrays.toString(mCV));
		System.out.println("scale " + scale);
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = true;
		String features = "hybrid";

		for(int split : splitCV) {
			String cls = String.valueOf(split);

			String classifierDir = simDir + "classifier/ml3_1slack/" + features + "_caffe_6_relu/";
			String inputDir = simDir + "Split_" + cls + "/files/";

			System.out.println("classifierDir: " + classifierDir + "\n");
			System.err.println("split " + split + "\t cls " + cls);

			boolean compute = false;
			for(double p : pCV) {
				for(int m : mCV) {
					for(double epsilon : epsilonCV) {
						for(double lambda : lambdaCV) {

							ML3CuttingPlane1Slack classifier = new ML3CuttingPlane1Slack(); 
							classifier.setLambda(lambda);
							classifier.setEpsilon(epsilon);
							classifier.setP(p);
							classifier.setM(m);
							classifier.setVerbose(1);
							classifier.setCpmax(cpmax);
							classifier.setCpmin(cpmin);

							String suffix = "_" + classifier.toString();
							File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
							if(fileClassifier == null) {
								compute = true;
							}
						}
					}
				}
			}

			if(compute || recompute) {
				List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(inputDir + "/multiclass_" + features + "_train_scale_" + scale + ".txt", numWords);
				List<STrainingSample<double[], Integer>> exampleTrain = new ArrayList<STrainingSample<double[], Integer>>();
				for(int i=0; i<listTrain.size(); i++) {
					exampleTrain.add(new STrainingSample<double[], Integer>(listTrain.get(i).sample.getFeature(0), listTrain.get(i).label));
				}

				List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(inputDir + "/multiclass_" + features + "_test_scale_" + scale + ".txt", numWords);
				List<STrainingSample<double[],Integer>> exampleTest = new ArrayList<STrainingSample<double[],Integer>>();
				for(int i=0; i<listTest.size(); i++) {
					exampleTest.add(new STrainingSample<double[], Integer>(listTest.get(i).sample.getFeature(0), listTest.get(i).label));
				}

				for(double p : pCV) {
					for(int m : mCV) {
						for(double epsilon : epsilonCV) {
							for(double lambda : lambdaCV) {

								ML3CuttingPlane1Slack classifier = new ML3CuttingPlane1Slack(); 
								classifier.setLambda(lambda);
								classifier.setEpsilon(epsilon);
								classifier.setP(p);
								classifier.setM(m);
								classifier.setVerbose(1);
								classifier.setCpmax(cpmax);
								classifier.setCpmin(cpmin);

								String suffix = "_" + classifier.toString();
								File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
								if(recompute || compute && fileClassifier == null) {
									classifier.train(exampleTrain);
									double acc = classifier.evaluation(exampleTrain);
									System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda + "\tp= " + p + "\tm= " + m);

									acc = classifier.evaluation(exampleTest);
									System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda + "\tp= " + p + "\tm= " + m);
									System.out.println("\n");

									fileClassifier = new File(classifierDir + "/" + cls + "/" + cls + "_" + scale + suffix + "_acc_" + acc + ".ser");
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
								}
							}
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
