package jstruct.data.voc2007.iccv15.mac.ranking;

import java.io.File;
import java.util.Arrays;

import jstruct.data.voc2007.VOC2007;
import fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingAPMantraM2CuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.util.VectorOp;

public class ClassifyRankingAPMantraCuttingPlane1Slack {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2007/";

	public static void main(String[] args) {

		//double[] lambdaCV = {1e-2, 1e-3, 1e-4, 1e-5};
		double[] lambdaCV = {1e-4};
		//double[] epsilonCV = {1e-2, 1e-3};
		double[] epsilonCV = {1e-3};
		int[] splitCV = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
		int[] scaleCV = {100,90,80,70,60,50,40,30};

		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		double[][] map = new double[scaleCV.length][splitCV.length];
		double[] mapAll = new double[scaleCV.length];

		for(int s=0; s<scaleCV.length; s++) {
			int scale = scaleCV[s];
			for(int split : splitCV) {
				String cls = VOC2007.getClasses()[split];	

				String classifierDir = simDir + "ICCV15/classifier/MantraCVPR/M2/AP/CuttingPlane1Slack/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						RankingAPMantraM2CuttingPlane1SlackBagImageRegion classifier = new RankingAPMantraM2CuttingPlane1SlackBagImageRegion();
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
							while(tm[n].compareTo("ap") != 0) {
								n++;
							}
							double ap = Double.parseDouble(tm[++n]);
							if(ap > map[s][split]) {
								map[s][split] = ap;
							}
						}
					}
				}
			}
			mapAll[s] = VectorOp.mean(map[s]);
		}

		for(int i=0; i<splitCV.length; i++) {
			for(int s=0; s<scaleCV.length; s++) {
				System.out.print(map[s][i] + " \t");
			}
			System.out.println();
		}

		System.out.println("map= " + VectorOp.mean(mapAll) + "\t" + Arrays.toString(mapAll));
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