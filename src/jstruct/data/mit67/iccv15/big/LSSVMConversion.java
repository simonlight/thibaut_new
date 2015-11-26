package jstruct.data.mit67.iccv15.big;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;

import fr.durandt.jstruct.latent.lssvm.multiclass.FastMulticlassLSSVMCuttingPlane1SlackBagImage;
import fr.durandt.jstruct.latent.lssvm.multiclass.FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion;

/**
 * Tests of LSSVM on UIUC Sports with deep features
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class LSSVMConversion {

	public static String simDir = "/home/durandt/simulation/MIT67/";

	public static void main(String[] args) {

		double[] lambdaCV = {1e-6};
		double[] epsilonCV = {1e-2};
		//Integer[] scaleCV = {100,90,80,70,60,50,40,30};
		Integer[] scaleCV = {Integer.parseInt(args[0])};
		//int[] splitCV = {1,2,3,4,5};
		int[] splitCV = {1};

		int cpmax = 500;
		int cpmin = 5;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		String features = "places";

		for(int scale : scaleCV) {
			for(int split : splitCV) {

				String cls = String.valueOf(split);

				String classifierDir = simDir + "/ICCV15/classifier/LSSVM/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/";
				String newClassifierDir = simDir + "/ICCV15/classifier/LSSVM/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegionC/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);


				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						FastMulticlassLSSVMCuttingPlane1SlackBagImage classifier = new FastMulticlassLSSVMCuttingPlane1SlackBagImage(); 
						classifier.setLambda(lambda);
						classifier.setEpsilon(epsilon);
						classifier.setCpmax(cpmax);
						classifier.setCpmin(cpmin);
						classifier.setVerbose(1);
						classifier.setnThreads(1);

						FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion newClassifier = new FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion(); 
						newClassifier.setLambda(lambda);
						newClassifier.setEpsilon(epsilon);
						newClassifier.setCpmax(cpmax);
						newClassifier.setCpmin(cpmin);
						newClassifier.setVerbose(1);
						newClassifier.setnThreads(1);

						String suffix = "_" + classifier.toString();
						File newFileClassifier = testPresenceFile(newClassifierDir + "/" + cls + "/", cls + "_" + scale + suffix);

						// si le fichier n'existe pas
						if(newFileClassifier == null) {

							File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);

							// load classifier
							ObjectInputStream ois;
							System.out.println("read classifier " + fileClassifier.getAbsolutePath());
							try {
								ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
								classifier = (FastMulticlassLSSVMCuttingPlane1SlackBagImage) ois.readObject();
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

							newClassifier.copy(classifier);
							
							String[] tmp = fileClassifier.getAbsolutePath().split("/");
							fileClassifier = new File(newClassifierDir + "/" + cls + "/" + tmp[tmp.length-1]);
							fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
							System.out.println("save classifier " + fileClassifier.getAbsolutePath());
							// save classifier
							ObjectOutputStream oos = null;
							try {
								oos = new ObjectOutputStream(new FileOutputStream(fileClassifier.getAbsolutePath()));
								oos.writeObject(newClassifier);
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
