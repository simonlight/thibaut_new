/**
 * 
 */
package jstruct.data.voc2011.action.mac;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import jstruct.data.voc2011.VOC2011;
import fr.durandt.jstruct.data.io.BagWriter;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class ReadBagKumar {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		int dim = 2405;
		int[] splitCV = {1,2,3,4,5};
		String set = "train";

		for(int i=0; i<VOC2011.getActionClasses().length; i++) {
			String cls = VOC2011.getActionClasses()[i];
			for(int split : splitCV) {

				File newFile = new File("/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr_2013_tutoriel/files/" + cls + "_" + split + "_" + set + ".txt");

				if(!newFile.exists()) {
					String file = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr_2013_tutoriel/" + set + "Files/" + cls + "_" + split + "_" + set + ".txt";
					List<STrainingSample<BagImageRegion, Integer>> bags = readKumarFiles(new File(file), dim);

					String pathImage = "/Volumes/Eclipse/LIP6/base/VOCdevkit/VOC2011/JPEGImages/";
					String pathFeature = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr_2013_tutoriel/features/";
					for(STrainingSample<BagImageRegion, Integer> bag : bags) {

						// change the name of the bag
						String name = bag.input.getName();
						String[] tmp = name.split("/");
						String imageName = tmp[tmp.length-1];
						bag.input.setName(pathImage + "/" + imageName + ".jpg");

						// change the path of the features
						for(int j=0; j<bag.input.numberOfInstances(); j++) {
							String fileInstance = bag.input.getInstanceFile(j);
							tmp = fileInstance.split("/");
							String instanceName = tmp[tmp.length-1];
							bag.input.setInstanceFile(j, pathFeature + "/" + imageName + "/" + instanceName);
						}
					}


					BagWriter.writeBagImageRegion(newFile, bags);
				}
			}
		}
		
		System.out.println("END");
	}


	public static List<STrainingSample<BagImageRegion, Integer>> readKumarFiles(File file, int dim) {

		int verbose = 0;
		List<STrainingSample<BagImageRegion, Integer>> list = null;

		try {
			// Read the file
			System.out.println("read file: " + file.getAbsolutePath());
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			// Initialize the list of bags
			list = new ArrayList<STrainingSample<BagImageRegion, Integer>>();
			int nbInstancesAll = 0;

			// Read the number of bags nbBags
			String ligne=br.readLine();
			int nbBags = Integer.parseInt(ligne);

			// Read nbBags bags
			for(int i=0; i<nbBags; i++) {
				System.out.print(".");
				if(i>0 && i % 100 == 0) System.out.print(i);

				// Read a new line
				ligne=br.readLine();
				if(verbose > 0) {
					System.out.println(i + " - read: " + ligne);
				}

				// Break the string in tokens
				StringTokenizer st = new StringTokenizer(ligne);

				// Read the name of the bag (usually the name of the image)
				String name = st.nextToken();
				// Read the number of instances
				int nbInstances = Integer.parseInt(st.nextToken());
				if(verbose>0) {
					System.out.println(i + " - name: " + name + "\tnbInstances: " + nbInstances);
				}

				// Create a new object BagImage and set the name
				BagImageRegion bag = new BagImageRegion();
				bag.setName(name);

				// Read nbInstances
				for(int j=0; j<nbInstances; j++) {

					int minX = Integer.parseInt(st.nextToken());
					int minY = Integer.parseInt(st.nextToken());
					int width = Integer.parseInt(st.nextToken());
					int height = Integer.parseInt(st.nextToken());

					Integer[] region = {minY, minX, minY+height, minX+width};
					bag.addRegion(region);

					int id = Integer.parseInt(st.nextToken());

					// read the annotation of each region
					// 0 -> meaning negative
					// -1 -> meaning unknown
					int annotation = Integer.parseInt(st.nextToken());

					// File of the (j+1)-th instance
					String fileInstance = name + "_" + id + ".feature";
					bag.addInstanceFile(fileInstance);
					bag.addInstance(readSparseFeature(new File(fileInstance), dim));
					if(verbose>0) {
						System.out.println(i + " - read instance " + j + "\t" + fileInstance);
					}

				}

				// Read the label of the bag
				int label = Integer.parseInt(st.nextToken());

				// Add the new bag to the list
				list.add(new STrainingSample<BagImageRegion, Integer>(bag, label));
				nbInstancesAll += bag.numberOfInstances();
			}
			br.close();
			System.out.println("\nnumber of bags= " + list.size() + "\tnumber of instances= " + nbInstancesAll + "\taverage number of instances per bag= " + ((double)nbInstancesAll/(double)list.size()));
		}
		catch (FileNotFoundException e) {
			System.out.println("File " + file.getAbsolutePath() + " not found");
			return null;
		}
		catch (IOException e) {
			System.out.println("Error parsing file " + file.getAbsolutePath());
			e.printStackTrace();
			return null;
		}

		return list;
	}

	public static double[] readSparseFeature(File file, int dim) {

		double[] feature = new double[dim];
		if(file.exists()) {

			try {
				InputStream ips = new FileInputStream(file); 
				InputStreamReader ipsr = new InputStreamReader(ips);
				BufferedReader br = new BufferedReader(ipsr);

				String ligne=br.readLine();

				// Break the string in tokens
				StringTokenizer st = new StringTokenizer(ligne);
				while(st.hasMoreTokens()) {
					String[] tmp = st.nextToken().split(":");
					int index = Integer.parseInt(tmp[0]);
					double val = Double.parseDouble(tmp[1]);
					//System.out.println(index + "\t" + val);
					feature[index-1] = val;
				}

				br.close();
			}
			catch (IOException e) {
				System.out.println("Error parsing file " + file);
			}
		}
		else {
			System.out.println("Features file " + file.getAbsolutePath() + " does not exist");
			System.exit(0);
		}

		return feature;

	}
}
