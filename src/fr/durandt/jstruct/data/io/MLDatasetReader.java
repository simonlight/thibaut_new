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
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

import fr.durandt.jstruct.util.VectorOp;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Read standard machine learning datasets
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class MLDatasetReader {

	public static List<TrainingSample<double[]>> readMulticlass(File file, int dim, int verbose) {
		return readMulticlass(file, dim, true, true, verbose);
	}

	public static List<TrainingSample<double[]>> readMulticlass(File file, int dim, boolean bias, boolean n2, int verbose) {

		System.out.print("read: " + file.getAbsolutePath());

		List<TrainingSample<double[]>> l = new ArrayList<TrainingSample<double[]>>();

		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			String ligne;
			while ((ligne=br.readLine()) != null){

				if(verbose > 1) {
					System.out.println(ligne);
				}
				StringTokenizer st = new StringTokenizer(ligne);

				int label = Integer.parseInt(st.nextToken());
				if(verbose > 1) {
					System.out.println("label " + label);
				}

				// read the feature
				double[] feature = new double[dim];
				while (st.hasMoreTokens()) { 
					String[] tmp = st.nextToken().split(":");
					feature[Integer.parseInt(tmp[0])-1] = Double.parseDouble(tmp[1]); 
				}
				if(verbose > 1) {
					System.out.println(l.size() + "\t" + Arrays.toString(feature));
				}

				l.add(new TrainingSample<double[]>(feature,label-1));
			}

			br.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file " + file);
		}

		for(TrainingSample<double[]> ts : l) {
			if(n2) {
				VectorOp.normL2(ts.sample);
			}
			if(bias) {
				ts.sample = VectorOp.addValeur(ts.sample,1);
			}
		}

		System.out.println("\t nb features= " + l.size() + "\tdim= " + dim);

		if(verbose > 0) {
			for(int i=0; i<l.size(); i++) {
				TrainingSample<double[]> ts = l.get(i);
				System.out.println(i + "/" + l.size() + "\t" + ts.label + "\t" + Arrays.toString(ts.sample));
			}
		}

		return l;
	}

	public static List<TrainingSample<double[]>> readBinary(File file, int dim, int verbose) {
		return readBinary(file, dim, true, true, verbose);
	}

	public static List<TrainingSample<double[]>> readBinary(File file, int dim, boolean bias, boolean n2, int verbose) {

		System.out.print("read: " + file.getAbsolutePath());

		List<TrainingSample<double[]>> l = new ArrayList<TrainingSample<double[]>>();

		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			String ligne;
			while ((ligne=br.readLine()) != null){
				if(!ligne.startsWith("#")) {
					if(verbose > 1) {
						System.out.println(ligne);
					}
					StringTokenizer st = new StringTokenizer(ligne);

					int label = Integer.parseInt(st.nextToken());
					if(verbose > 1) {
						System.out.println("label " + label);
					}

					// Read feature
					double[] feature = new double[dim];
					while (st.hasMoreTokens()) { 
						String[] tmp = st.nextToken().split(":");
						feature[Integer.parseInt(tmp[0])-1] = Double.parseDouble(tmp[1]); 
					}
					if(verbose > 1) {
						System.out.println(l.size() + "\t" + Arrays.toString(feature));
					}

					l.add(new TrainingSample<double[]>(feature, label));
				}
			}

			br.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file " + file);
		}

		for(TrainingSample<double[]> ts : l) {
			// L2 normalization of the features
			if(n2) {
				VectorOp.normL2(ts.sample);
			}
			// 
			if(bias) {
				ts.sample = VectorOp.addValeur(ts.sample,1);
			}
		}

		System.out.println("\t nb features= " + l.size() + "\tdim= " + dim);

		if(verbose > 0) {
			for(int i=0; i<l.size(); i++) {
				TrainingSample<double[]> ts = l.get(i);
				System.out.println(i + "/" + l.size() + "\t" + ts.label + "\t" + Arrays.toString(ts.sample));
			}
		}

		return l;
	}

}
