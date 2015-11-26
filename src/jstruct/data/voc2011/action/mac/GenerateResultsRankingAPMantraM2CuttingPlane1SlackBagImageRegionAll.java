/**
 * 
 */
package jstruct.data.voc2011.action.mac;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import jstruct.data.voc2011.VOC2011;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.data.io.VocData;
import fr.durandt.jstruct.data.io.VocReader;
import fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingAPMantraM2CuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class GenerateResultsRankingAPMantraM2CuttingPlane1SlackBagImageRegionAll {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr2014/";

	private static int numWords = 2405;

	public static void main(String[] args) {

		double[] lambdaCls = {0.01, 0.01, 0.1, 0.1, 1.0E-6, 1.0E-6, 1.0E-6, 1.0E-6, 0.1, 1.0E-6};
		double epsilon = 1e-3;

		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		for(int iCls=0; iCls<VOC2011.getActionClasses().length; iCls++) {
			//for(int iCls=0; iCls<1; iCls++) {

			double lambda = lambdaCls[iCls];

			String cls = VOC2011.getActionClasses()[iCls];		

			String classifierDir = simDir + "/ICCV15/classifier/Mantra/M2/AP/";
			String scoresDir = simDir + "/results/MANTRA/CV/results/VOC2011/Action/";
			String inputDir = simDir + "/files/";

			System.out.println("classifierDir: " + classifierDir + "\n");

			String filename = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr2014/VOCdevkit/VOC2011/ImageSets/Action/" + cls + "_test.txt";
			List<VocData> data = VocReader.readActionTest(new File(filename));

			RankingAPMantraM2CuttingPlane1SlackBagImageRegion classifier = new RankingAPMantraM2CuttingPlane1SlackBagImageRegion();
			classifier.setLambda(lambda);
			classifier.setEpsilon(epsilon);
			classifier.setCpmax(cpmax);
			classifier.setCpmin(cpmin);
			classifier.setVerbose(1);
			classifier.setOptim(optim);

			String suffix = "_" + classifier.toString();
			File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + suffix);

			// load classifier
			ObjectInputStream ois;
			System.out.println("read classifier " + fileClassifier.getAbsolutePath());
			try {
				ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
				classifier = (RankingAPMantraM2CuttingPlane1SlackBagImageRegion) ois.readObject();
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

			List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/" + cls + "_test.txt", numWords, true, true, null, true, 0);

			List<Double> scores = new ArrayList<Double>(listTest.size());
			for(STrainingSample<BagImageRegion, Integer> ts : listTest) {
				scores.add(classifier.valueOf(ts.input));
			}
			
			File file = new File(scoresDir + "/comp10_action_test_" + cls + ".txt");
			writeScores(file, listTest, scores, data);

			System.out.println("\n");
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

	public static void writeScores(File file, List<STrainingSample<BagImageRegion, Integer>> list, List<Double> scores, List<VocData> data) {

		System.out.println("Write scores file " + file.getAbsolutePath());

		// Create the directory if not exist
		file.getAbsoluteFile().getParentFile().mkdirs();

		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			for(int i=0; i<list.size(); i++) {
				String[] tmp = list.get(i).input.getName().split("/");
				tmp = tmp[tmp.length-1].split(".jpg");

				if(tmp[0].compareTo(data.get(i).getName()) != 0) {
					System.out.println(tmp[0]);
					System.out.println(data.get(i).getName());
					System.exit(0);
				}

				bw.write(data.get(i).getName() + "\t" + data.get(i).getIndexRegionAction() + "\t" + scores.get(i));
				bw.write("\n");
			}

			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			e.printStackTrace();
			return;
		}

	}

}
