package jstruct.data.uiuc.iccv15.mac;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReaderXML;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.mantra.nips15.multiclass.MulticlassMantraBetaPegasosBagImageRegion;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * Tests of LSSVM on UIUC Sports with deep features
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class TestMulticlassMantraBetaPegasosBagImageXML {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/UIUCSports/";

	public static void main(String[] args) {

		int numWords = 4096;

		//double[] lambdaCV = {1e-4, 1e-5, 1e-6, 1e-7};
		double[] lambdaCV = {1e-6};
		double[] epsilonCV = {1e-2};
		//int[] scaleCV = {100,90,80,70,60,50,40,30};
		int[] scaleCV = {90};
		//int[] splitCV = {1,2,3,4,5};
		int[] splitCV = {1};

		int maxIter = 50;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = true;
		//String features = "caffe_vgg19_6_relu_vd_multiscale_256_384_512";
		String features = "caffe_hybrid_caffe_6_relu";

		for(int scale : scaleCV) {
			for(int split : splitCV) {

				String cls = String.valueOf(split);

				String classifierDir = simDir + "/ICCV15/classifier/MANTRA/Minus/CuttingPlane1Slack/Multiclass/Fast/" + features + "/BagImageRegion/";
				String predictionDir = simDir + "/ICCV15/prediction/MANTRA/Minus/CuttingPlane1Slack/Multiclass/Fast/" + features + "/BagImageRegion/";
				String scoreDir = simDir + "/ICCV15/scores/MANTRA/Minus/CuttingPlane1Slack/Multiclass/Fast/" + features + "/BagImageRegion/";
				String inputDir = simDir + "Split_" + cls + "/files_BagImageRegion/" + features + "/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

				boolean compute = false;
				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						MulticlassMantraBetaPegasosBagImageRegion classifier = new MulticlassMantraBetaPegasosBagImageRegion(); 
						classifier.setLambda(lambda);
						classifier.setMaxIter(maxIter);
						classifier.setVerbose(1);

						String suffix = "_" + classifier.toString();
						File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
						if(fileClassifier == null) {
							compute = true;
						}

						File filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + scale + suffix + "_train.txt");
						if(!filePrediction.exists()) {
							compute = true;
						}

						filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + scale + suffix + "_test.txt");
						if(!filePrediction.exists()) {
							compute = true;
						}	

						File fileScores = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
						if(!fileScores.exists()) {
							compute = true;
						}	

						fileScores = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
						if(!fileScores.exists()) {
							compute = true;
						}	
					}
				}

				if(compute || recompute) {
					List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReaderXML.readBagImageRegion(new File(inputDir + "/multiclass_train_scale_" + scale + ".xml"), numWords, true, true, null, true, 0);
					List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>>();
					for(int i=0; i<listTrain.size(); i++) {
						exampleTrain.add(new STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>(new LatentRepresentation<BagImageRegion, Integer>(listTrain.get(i).input,0), listTrain.get(i).output));
					}

					List<STrainingSample<BagImageRegion, Integer>> listTest = BagReaderXML.readBagImageRegion(new File(inputDir + "/multiclass_test_scale_" + scale + ".xml"), numWords, true, true, null, true, 0);
					List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>>();
					for(int i=0; i<listTest.size(); i++) {
						exampleTest.add(new STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>(new LatentRepresentation<BagImageRegion, Integer>(listTest.get(i).input,0), listTest.get(i).output));
					}

					for(double epsilon : epsilonCV) {
						for(double lambda : lambdaCV) {

							MulticlassMantraBetaPegasosBagImageRegion classifier = new MulticlassMantraBetaPegasosBagImageRegion(); 
							classifier.setLambda(lambda);
							classifier.setMaxIter(maxIter);
							classifier.setVerbose(1);

							String suffix = "_" + classifier.toString();
							File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
							if(recompute || compute && fileClassifier == null) {
								classifier.train(exampleTrain);
								double acc = classifier.accuracy(exampleTrain);
								System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );

								acc = classifier.accuracy(exampleTest);
								System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );
								System.out.println("\n");

								/*fileClassifier = new File(classifierDir + "/" + cls + "/" + cls + "_" + scale + suffix + "_acc_" + acc + ".ser");
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
								}*/
							}
							/*else {
								// load classifier
								ObjectInputStream ois;
								System.out.println("read classifier " + fileClassifier.getAbsolutePath());
								try {
									ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
									classifier = (FastMulticlassMantraCuttingPlane1SlackBagImageRegion) ois.readObject();
								} 
								catch (FileNotFoundException e) {
									e.printStackTrace();
								} 
								catch (IOException e) {
									e.printStackTrace();
								} 
								catch (ClassNotFoundException e) {
									e.printStackTrace();
								}

								double acc = classifier.accuracy(exampleTest);
								System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );
								System.out.println("\n");
							}

							// prediction
							File filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + scale + suffix + "_train.txt");
							if(!filePrediction.exists()) {
								classifier.writePrediction(filePrediction, exampleTrain);
							}

							filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + scale + suffix + "_test.txt");
							if(!filePrediction.exists()) {
								classifier.writePrediction(filePrediction, exampleTest);
							}

							// scores
							File fileScores = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
							if(!fileScores.exists()) {
								classifier.writeScores(fileScores, exampleTrain);
							}

							fileScores = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
							if(!fileScores.exists()) {
								classifier.writeScores(fileScores, exampleTest);
							}*/
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
