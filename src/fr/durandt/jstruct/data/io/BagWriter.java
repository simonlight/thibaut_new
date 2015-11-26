/**
 * 
 */
package fr.durandt.jstruct.data.io;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;
import fr.durandt.jstruct.variable.BagLabel;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class BagWriter {

	public static void writeBagLabel(File file, List<TrainingSample<LatentRepresentation<BagLabel,Integer>>> list) {

		System.out.println("Write BagLabel file " + file.getAbsolutePath());

		// Create the directory if not exist
		file.getAbsoluteFile().getParentFile().mkdirs();

		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			int n=0;
			for(int i=0; i<list.size(); i++) {
				for(int h=0; h<list.get(i).sample.x.numberOfInstances(); h++) {
					// Index of the features
					bw.write(n + ":");
					// Index of the bags
					bw.write(i + ":");
					// Label of the instance
					bw.write(list.get(i).sample.x.getLabel(h) + "\t");

					// Write the vector represenation
					double[] instance = list.get(i).sample.x.getInstance(h);
					for(int d=0; d<instance.length; d++) {
						bw.write((d+1) + ":" + instance[d] + "\t");
					}
					bw.write("\n");
					n++;
				}
			}

			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			e.printStackTrace();
			return;
		}

	}

	public static void writeBagImageRegion(File file, List<STrainingSample<BagImageRegion, Integer>> list) {

		System.out.println("Write BagImageRegion file " + file.getAbsolutePath());

		// Create the directory if not exist
		file.getAbsoluteFile().getParentFile().mkdirs();

		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			bw.write(list.size() + "\n");
			for(int i=0; i<list.size(); i++) {
				// Write the name of the bag
				bw.write(list.get(i).input.getName());

				// Write the label of the bag
				bw.write("\t" + list.get(i).output);

				// Write the number of instance per bag
				bw.write("\t" + list.get(i).input.numberOfInstances());

				//
				for(int j=0; j<list.get(i).input.numberOfInstances(); j++) {
					// Write the file of the (j+1)-th instance
					bw.write("\t" + list.get(i).input.getInstanceFile(j));

					// Write feature if not exist
					writeFeature(list.get(i).input.getInstanceFile(j), list.get(i).input.getInstance(j));

					// Write the position of the region
					bw.write("\t" + list.get(i).input.getRegion(j)[0]);
					bw.write("\t" + list.get(i).input.getRegion(j)[1]);
					bw.write("\t" + list.get(i).input.getRegion(j)[2]);
					bw.write("\t" + list.get(i).input.getRegion(j)[3]);
				}
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

	private static void writeFeature(String file, double[] feature) {
		writeFeature(new File(file), feature);
	}

	private static void writeFeature(File file, double[] feature) {
		if(!file.exists()) {
			file.getAbsoluteFile().getParentFile().mkdirs();
			try {
				OutputStream ops = new FileOutputStream(file); 
				OutputStreamWriter opsr = new OutputStreamWriter(ops);
				BufferedWriter bw = new BufferedWriter(opsr);

				for(int i=0; i<feature.length; i++) {
					bw.write(feature[i] + "\n");
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

}
