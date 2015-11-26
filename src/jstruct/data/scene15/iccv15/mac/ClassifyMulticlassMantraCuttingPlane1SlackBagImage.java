package jstruct.data.scene15.iccv15.mac;

import java.io.File;
import java.util.Arrays;

import fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.FastMulticlassMantraCVPRCuttingPlane1SlackBagImage;
import fr.durandt.jstruct.util.VectorOp;

/**
 * Tests of LSSVM on UIUC Sports with deep features
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class ClassifyMulticlassMantraCuttingPlane1SlackBagImage {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/15scenes/";

	public static void main(String[] args) {

		double[] lambdaCV = {1e-6};
		double[] epsilonCV = {1e-2};
		Integer[] scaleCV = {100,90,80,70,60,50};
		int[] splitCV = {1,2,3,4,5};
		//int[] splitCV = {1};

		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		String features = "places";

		double[][] accuracy = new double[scaleCV.length][splitCV.length];
		double[] accuracyAll = new double[scaleCV.length];

		for(int s=0; s<scaleCV.length; s++) {
			int scale = scaleCV[s];
			for(int split : splitCV) {

				String cls = String.valueOf(split);

				String classifierDir = simDir + "/ICCV15/classifier/MANTRA/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/";
				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

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
						if(fileClassifier != null) {
							String[] tm = fileClassifier.getAbsolutePath().split(".ser")[0].split("_");
							int n=0;
							while(tm[n].compareTo("acc") != 0) {
								n++;
							}
							double acc = Double.parseDouble(tm[++n]);
							if(acc > accuracy[s][split-1]) {
								accuracy[s][split-1] = acc;
							}
						}
					}
				}
			}
			accuracyAll[s] = VectorOp.mean(accuracy[s]);
		}

		for(int s=0; s<scaleCV.length; s++) {
			System.out.println(Arrays.toString(accuracy[s]));
		}

		System.out.print("accuracy= ");
		for(double d : accuracyAll) {
			System.out.print(d + "\t");
		}
		System.out.println();
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
