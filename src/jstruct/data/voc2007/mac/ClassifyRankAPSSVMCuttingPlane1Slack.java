package jstruct.data.voc2007.mac;

import java.io.File;
import java.util.Arrays;

import jstruct.data.voc2007.VOC2007;
import fr.durandt.jstruct.ssvm.ranking.DoubleRankAPSSVMCuttingPlane1Slack;
import fr.durandt.jstruct.util.VectorOp;

public class ClassifyRankAPSSVMCuttingPlane1Slack {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2007/";

	private static int numWords = 2048;

	public static void main(String[] args) {

		double[] lambdaCV = {1e-3, 1e-4, 1e-5, 1e-6, 1e-8};
		double[] epsilonCV = {1e-2,1e-3};
		int[] scaleCV = {100};
		int[] splitCV = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
		//int[] splitCV = {0};
		int scale = 100;

		int cpmax = 500;
		int cpmin = 5;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		double[] map = new double[splitCV.length];

		for(int i=0; i<splitCV.length; i++) {
			int split = splitCV[i];
			String cls = VOC2007.getClasses()[split];

			String classifierDir = simDir + "classifier/ssvm_ranking/AP/matconvnet_m_" + numWords + "_layer_20/";

			System.out.println("classifierDir: " + classifierDir + "\n");
			System.err.println("split " + split + "\t cls " + cls);

			for(double epsilon : epsilonCV) {
				for(double lambda : lambdaCV) {

					DoubleRankAPSSVMCuttingPlane1Slack classifier = new DoubleRankAPSSVMCuttingPlane1Slack(); 
					classifier.setLambda(lambda);
					classifier.setEpsilon(epsilon);
					classifier.setCpmax(cpmax);
					classifier.setCpmin(cpmin);
					classifier.setVerbose(1);

					String suffix = "_" + classifier.toString();
					File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
					if(fileClassifier != null) {
						String[] tm = fileClassifier.getAbsolutePath().split(".ser")[0].split("_");
						int n=0;
						while(tm[n].compareTo("ap") != 0) {
							n++;
						}
						double ap = Double.parseDouble(tm[++n]);
						if(ap > map[i]) {
							map[i] = ap;
						}
					}
				}
			}
		}
		
		System.out.println("map= " + VectorOp.mean(map) + "\t" + Arrays.toString(map));
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