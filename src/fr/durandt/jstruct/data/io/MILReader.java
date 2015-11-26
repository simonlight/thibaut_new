/**
 * 
 */
package fr.durandt.jstruct.data.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.Bag;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class MILReader {
	
	public static List<TrainingSample<Bag>> read(File file) {
		return read(file, true, true);
	}

	public static List<TrainingSample<Bag>> read(File file, boolean bias, boolean norm2) {

		System.out.print("read: " + file.getAbsolutePath());

		List<TrainingSample<Bag>> l = new ArrayList<TrainingSample<Bag>>();

		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			String ligne;
			ligne=br.readLine(); // first line of the file : "#Generated by milsample."

			while ((ligne=br.readLine()) != null){
				StringTokenizer st = new StringTokenizer(ligne);
				ligne = st.nextToken();
				String[] tmp = ligne.split(":");

				// index of the features
				// int sample = Integer.parseInt(tmp[0]);
				// index of the bag
				int bag = Integer.parseInt(tmp[1]);
				// label of the bag
				int label = Integer.parseInt(tmp[2]);

				// if the bag does not exist, create a new bag
				if(bag>=l.size()) {
					l.add(bag, new TrainingSample<Bag>(new Bag(), label));
					l.get(bag).sample.setName(String.valueOf(bag));
				}
				//System.out.println("sample: " + sample + "\tbag: " + bag + "\tlabel: " + label);

				// read the feature
				List<Double> f = new ArrayList<Double>();
				while (st.hasMoreTokens()) { 
					tmp = st.nextToken().split(":");
					f.add(Double.parseDouble(tmp[1])); 
				}
				//System.out.println("size: " + f.size());

				double[] feature = new double[f.size()];
				for(int i=0; i<f.size(); i++) {
					feature[i] = f.get(i);
				}
				l.get(bag).sample.addInstance(feature);
			}

			br.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file " + file);
		}
		System.out.println("\t nb bags= " + l.size());
		
		for(TrainingSample<Bag> ts : l) {
			for(int i=0; i<ts.sample.getInstances().size(); i++) {
				// L2 normalization 
				if(norm2) {
					VectorOp.normL2(ts.sample.getInstance(i));
				}
				// add a constant 1 feature for the bias
				if(bias) {
					ts.sample.setInstance(i,VectorOp.addValeur(ts.sample.getInstance(i),1));
				}
			}
		}

		/*for(TrainingSample<Bag> ts : l) {
			System.out.println(ts.sample);
		}*/

		return l;
	}

}
