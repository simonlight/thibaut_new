package jstruct.data.ppmi.iccv15.big;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPRCuttingPlane1SlackBagImage;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImage;

/**
 * Tests of LSSVM on UIUC Sports with deep features
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class TestMulticlassMantraCuttingPlane1SlackBagImage {

	public static String simDir = "/home/durandt/simulation/PPMI/";

	public static void main(String[] args) {

		int numWords = 4096;

		double[] lambdaCV = {1e-6};
		double[] epsilonCV = {1e-2};
		Integer[] scaleCV = {100,90,80,70,60,50,40,30};
		int[] splitCV = {1};

		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = false;
		String features = "imagenet";

		for(int scale : scaleCV) {
			for(int split : splitCV) {

				String cls = "multiclass";

				String classifierDir = simDir + "/ICCV15/classifier/MANTRA/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/";
				String inputDir = simDir + "/files_BagImage/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

				boolean compute = false;
				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						FastMulticlassMantraCVPRCuttingPlane1SlackBagImage classifier = new FastMulticlassMantraCVPRCuttingPlane1SlackBagImage(); 
						classifier.setLambda(lambda);
						classifier.setEpsilon(epsilon);
						classifier.setCpmax(cpmax);
						classifier.setCpmin(cpmin);
						classifier.setVerbose(1);
						classifier.setOptim(optim);

						String suffix = "_" + classifier.toString();
						File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
						if(fileClassifier == null) {
							compute = true;
						}

					}
				}

				if(compute || recompute) {
					List<STrainingSample<BagImage, Integer>> listTrain = BagReader.readBagImage(inputDir + "/" + cls + "_" + features + "_train_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
					List<STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>>();
					for(int i=0; i<listTrain.size(); i++) {
						exampleTrain.add(new STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>(new LatentRepresentation<BagImage, Integer>(listTrain.get(i).input,0), listTrain.get(i).output));
					}

					List<STrainingSample<BagImage, Integer>> listTest = BagReader.readBagImage(inputDir + "/" + cls + "_" + features + "_test_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
					List<STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>>();
					for(int i=0; i<listTest.size(); i++) {
						exampleTest.add(new STrainingSample<LatentRepresentation<BagImage, Integer>,Integer>(new LatentRepresentation<BagImage, Integer>(listTest.get(i).input,0), listTest.get(i).output));
					}

					for(double epsilon : epsilonCV) {
						for(double lambda : lambdaCV) {

							FastMulticlassMantraCVPRCuttingPlane1SlackBagImage classifier = new FastMulticlassMantraCVPRCuttingPlane1SlackBagImage(); 
							classifier.setLambda(lambda);
							classifier.setEpsilon(epsilon);
							classifier.setCpmax(cpmax);
							classifier.setCpmin(cpmin);
							classifier.setVerbose(1);
							classifier.setnThreads(1);
							classifier.setOptim(optim);

							String suffix = "_" + classifier.toString();
							File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
							if(recompute || compute && fileClassifier == null) {
								classifier.train(exampleTrain);
								double acc = classifier.accuracy(exampleTrain);
								System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );

								acc = classifier.accuracy(exampleTest);
								System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );
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
