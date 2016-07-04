package fr.durandt.jstruct.data.io;

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

import fr.durandt.jstruct.extern.pca.PrincipalComponentAnalysis;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.BagImage;
import fr.durandt.jstruct.variable.BagImageRegion;
import fr.durandt.jstruct.variable.BagImageSeg;
import fr.durandt.jstruct.variable.BagLabel;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class BagReader {

	/*public static List<TrainingSample<BagMIL>> readBagMIL(String file, int dim) {
		List<TrainingSample<BagMIL>> list = readBagMIL(new File(file), dim, true, null);
		return list;
	}

	public static List<TrainingSample<BagMIL>> readBagMIL(String file, int dim, boolean bias, PrincipalComponentAnalysis pca) {
		List<TrainingSample<BagMIL>> list = readBagMIL(new File(file), dim, bias, pca);
		return list;
	}

	public static List<TrainingSample<BagMIL>> readBagMIL(File file, int dim, boolean bias, PrincipalComponentAnalysis pca) {
		List<TrainingSample<BagMIL>> list = null;
		if(file.exists()) {
			try {
				System.out.println("read bag: " + file.getAbsolutePath() + "\tdim= " + dim);
				InputStream ips = new FileInputStream(file); 
				InputStreamReader ipsr = new InputStreamReader(ips);
				BufferedReader br = new BufferedReader(ipsr);

				list = new ArrayList<TrainingSample<BagMIL>>();
				int nbInstancesAll = 0;

				String ligne;
				ligne=br.readLine();
				int nbBag = Integer.parseInt(ligne);
				for(int i=0; i<nbBag; i++) {
					System.out.print(".");
					if(i>0 && i % 100 == 0) System.out.print(i);
					ligne=br.readLine();
					//System.out.println(ligne);
					StringTokenizer st = new StringTokenizer(ligne);
					String name = st.nextToken();
					int label = Integer.parseInt(st.nextToken());
					int nbInstances = Integer.parseInt(st.nextToken());
					//System.out.println("name: " + name + "\tlabel: " + label + "\tnbInstances: " + nbInstances);
					BagMIL bag = new BagMIL();
					bag.setName(name);
					for(int j=0; j<nbInstances; j++) {
						String filefeature = st.nextToken();
						bag.addFileFeature(filefeature);
						double[] feature = readFeature(new File(filefeature));
						if(feature.length != dim) {
							System.out.println("ERROR features - dim= " + feature.length + " != " + dim);
							System.out.println("file " + filefeature);
							feature = null;
							System.exit(0);
						}
						bag.addFeature(feature);
					}
					nbInstancesAll += bag.getFeatures().size();
					list.add(new TrainingSample<BagMIL>(bag,label));
				}
				br.close();
				System.out.println("\nnb bags: " + list.size() + "\tnb instances: " + nbInstancesAll + "\tnb moyen instances: " + (nbInstancesAll/list.size()));

				for(TrainingSample<BagMIL> ts : list) {
					for(int i=0; i<ts.sample.getFeatures().size(); i++) {
						VectorOp.normL2(ts.sample.getFeature(i));
						if(pca != null) ts.sample.setFeature(i, pca.sampleToEigenSpace(ts.sample.getFeature(i)));
						// add a constant 1 feature for the bias
						if(bias) ts.sample.setFeature(i,VectorOp.addValeur(ts.sample.getFeature(i),1));
					}
				}

			}
			catch (IOException e) {
				System.out.println("Error parsing file " + file);
				return null;
			}
		}
		else {
			System.out.println("file " + file.getAbsolutePath() + " does not exist");
		}
		return list;
	}*/

	private static final int LatentRepresentation = 0;

	private static double[] readFeature(File file) {

		double[] feature = null;
		if(file.exists()) {
			List<Double> l = new ArrayList<Double>();
			String ligne = null;
			try {
				InputStream ips = new FileInputStream(file); 
				InputStreamReader ipsr = new InputStreamReader(ips);
				BufferedReader br = new BufferedReader(ipsr);

				while ((ligne=br.readLine()) != null){
					l.add(Double.parseDouble(ligne));
				}

				br.close();
			}
			catch (IOException e) {
				System.out.println("Error parsing file " + file);
				e.printStackTrace();
				System.exit(0);
			}
			catch (NumberFormatException e) {
				System.out.println("Error parsing file " + file);
				System.out.println("ligne= " + ligne + "\tl= " + l.size());
				e.printStackTrace();
				System.exit(0);
			}

			feature = new double[l.size()];
			for(int i=0; i<l.size(); i++) {
				feature[i] = l.get(i);
			}

			//System.out.println("PPMI - read feature: " + file.getAbsoluteFile() + "\tdim: " + feature.length);
		}
		else {
			System.out.println("Features file " + file.getAbsolutePath() + " does not exist");
			System.exit(0);
		}

		return feature;

	}

	public static List<STrainingSample<BagImage, Integer>> readBagImageNoFeatures(File file, int dim) {
		return readBagImage(file, dim, false, false, null, false, 0);
	}

	public static List<STrainingSample<BagImage, Integer>> readBagImageNoFeatures(String file, int dim) {
		return readBagImageNoFeatures(new File(file), dim);
	}

	public static List<STrainingSample<BagImage, Integer>> readBagImage(String file, int dim) {
		return readBagImage(new File(file), dim);
	}

	public static List<STrainingSample<BagImage, Integer>> readBagImage(File file, int dim) {
		return readBagImage(file, dim, false, false, null, true, 0);
	}

	public static List<STrainingSample<BagImage, Integer>> readBagImage(File file, int dim, int verbose) {
		return readBagImage(file, dim, false, false, null, true, verbose);
	}

	public static List<STrainingSample<BagImage, Integer>> readBagImage(String file, int dim, int verbose) {
		return readBagImage(new File(file), dim, verbose);
	}

	public static List<STrainingSample<BagImage, Integer>> readBagImage(String file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose) {
		return readBagImage(new File(file), dim, norm2, bias, pca, withFeatures, verbose);
	}
	
	public static List<TrainingSample<LatentRepresentation<BagLabel,Integer>>> readBagLabel(String file, int dim, boolean bias, boolean norm2, int verbose) {
		return readBagLabel(new File(file), dim, bias, norm2, verbose);
	}
	
	public static List<TrainingSample<LatentRepresentation<BagImage,Integer>>> readBagImageLatent(String file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose, String dataSource) {
		return readBagImageLatent(new File(file), dim, norm2, bias, pca, withFeatures, verbose, dataSource);
	}
		

	/**
	 * The number of instance per bag can change for each bag. <br/>
	 * &lt nbBags= number of bags &gt <br/>
	 * &lt name &gt &lt label &gt &lt nbIntances = number of instances &gt &lt file of instance 0 &gt ... &lt file of instance nbIntances-1 &gt <br/>
	 * &lt name &gt &lt label &gt &lt nbIntances = number of instances &gt &lt file of instance 0 &gt ... &lt file of instance nbIntances-1 &gt <br/>
	 * ... <br/>
	 * @param file
	 * @param dim
	 * @param norm2
	 * @param bias
	 * @param pca
	 * @return 
	 * The method return null if the file does not exist.
	 */
	public static List<STrainingSample<BagImage, Integer>> readBagImage(File file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose) {
		List<STrainingSample<BagImage, Integer>> list = null;
		try {
			// Read the file
			System.out.println("read bag: " + file.getAbsolutePath() + "\tdim= " + dim);
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			// Initialize the list of bags
			list = new ArrayList<STrainingSample<BagImage, Integer>>();
			int nbInstancesAll = 0;

			// Read the number of bags nbBags
			String ligne=br.readLine();
			int nbBags = Integer.parseInt(ligne);
			//test!!!!!!!!!!!!!
			nbBags=2;
			//
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
				
				// Read the label of the bag
				int label = Integer.parseInt(st.nextToken());
				// Read the number of instances
				int nbInstances = Integer.parseInt(st.nextToken());
				if(verbose>0) {
					System.out.println(i + " - name: " + name + "\tlabel: " + label + "\tnbInstances: " + nbInstances);
				}

				// Create a new object BagImage and set the name
				BagImage bag = new BagImage();
				bag.setName(name);

				// Read nbInstances
				for(int j=0; j<nbInstances; j++) {
					// Read the file of the (j+1)-th instance
					String fileInstance = st.nextToken();
					fileInstance =fileInstance.replace("home", "local");
					fileInstance =fileInstance.replace("matconvnet_m_2048_features", "m_2048_trainval_features");
					bag.addInstanceFile(fileInstance);
					if(verbose>0) {
						System.out.println(i + " - read instance " + j + "\t" + fileInstance);
					}

					if(withFeatures) {
						// Read the feature in the given file
						double[] feature = readFeature(new File(fileInstance));
						if(feature.length != dim) {
							System.out.println("ERROR read features - dim= " + feature.length + " != " + dim);
							System.out.println("file " + fileInstance);
							feature = null;
							System.exit(0);
						}
						// Add the feature to the bag
						bag.addInstance(feature);
					}
				}
				// Add the new bag to the list
				list.add(new STrainingSample<BagImage, Integer>(bag, label));
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

		// Pre-treatment of the instances
		for(STrainingSample<BagImage, Integer> ts : list) {
			for(int i=0; i<ts.input.numberOfInstances(); i++) {
				// L2 normalization of the instance
				if(norm2) {
					VectorOp.normL2(ts.input.getInstance(i));
				}
				// PCA
				if(pca != null) {
					ts.input.setInstance(i, pca.sampleToEigenSpace(ts.input.getInstance(i)));
				}
				// Add a constant 1 feature for the bias
				if(bias) {
					ts.input.setInstance(i,VectorOp.addValeur(ts.input.getInstance(i),1));
				}
			}
		}

		return list;
	}

	public static List<STrainingSample<BagImageRegion, Integer>> readBagImageRegion(String file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose) {
		return readBagImageRegion(new File(file), dim, norm2, bias, pca, withFeatures, verbose);
	}

	/**
	 * The number of instance per bag can change for each bag. <br/>
	 * &lt nbBags= number of bags &gt <br/>
	 * &lt name &gt &lt label &gt &lt nbIntances = number of instances &gt &lt file of instance 0 &gt &lt x1 &gt &lt y1 &gt &lt x2 &gt &lt y2 &gt &lt file of instance 1 &gt &lt x1 &gt &lt y1 &gt &lt x2 &gt &lt y2 &gt ... &lt file of instance nbIntances-1 &gt &lt x1 &gt &lt y1 &gt &lt x2 &gt &lt y2 &gt <br/>
	 * &lt name &gt &lt label &gt &lt nbIntances = number of instances &gt &lt file of instance 0 &gt &lt x1 &gt &lt y1 &gt &lt x2 &gt &lt y2 &gt &lt file of instance 1 &gt &lt x1 &gt &lt y1 &gt &lt x2 &gt &lt y2 &gt ... &lt file of instance nbIntances-1 &gt &lt x1 &gt &lt y1 &gt &lt x2 &gt &lt y2 &gt <br/>
	 * ... <br/>
	 * @param file
	 * @param dim
	 * @param norm2
	 * @param bias
	 * @param pca
	 * @return 
	 * The method return null if the file does not exist.
	 */
	public static List<STrainingSample<BagImageRegion, Integer>> readBagImageRegion(File file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose) {
		List<STrainingSample<BagImageRegion, Integer>> list = null;
		try {
			// Read the file
			System.out.println("read bag: " + file.getAbsolutePath() + "\tdim= " + dim);
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
				// Read the label of the bag
				int label = Integer.parseInt(st.nextToken());
				// Read the number of instances
				int nbInstances = Integer.parseInt(st.nextToken());
				if(verbose>0) {
					System.out.println(i + " - name: " + name + "\tlabel: " + label + "\tnbInstances: " + nbInstances);
				}

				// Create a new object BagImage and set the name
				BagImageRegion bag = new BagImageRegion();
				bag.setName(name);

				// Read nbInstances
				for(int j=0; j<nbInstances; j++) {
					// Read the file of the (j+1)-th instance
					String fileInstance = st.nextToken();
					bag.addInstanceFile(fileInstance);
					if(verbose>0) {
						System.out.println(i + " - read instance " + j + "\t" + fileInstance);
					}

					if(withFeatures) {
						// Read the feature in the given file
						double[] feature = readFeature(new File(fileInstance));
						if(feature.length != dim) {
							System.out.println("ERROR read features - dim= " + feature.length + " != " + dim);
							System.out.println("file " + fileInstance);
							feature = null;
							System.exit(0);
						}
						// Add the feature to the bag
						bag.addInstance(feature);
					}

					// Read the coordinates of the regions
					int x1 = Integer.parseInt(st.nextToken());
					int y1 = Integer.parseInt(st.nextToken());
					int x2 = Integer.parseInt(st.nextToken());
					int y2 = Integer.parseInt(st.nextToken());
					Integer[] region = {x1, y1, x2, y2};
					bag.addRegion(region);

				}
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

		// Pre-treatment of the instances
		for(STrainingSample<BagImageRegion, Integer> ts : list) {
			for(int i=0; i<ts.input.numberOfInstances(); i++) {
				// L2 normalization of the instance
				if(norm2) {
					VectorOp.normL2(ts.input.getInstance(i));
				}
				// PCA
				if(pca != null) {
					ts.input.setInstance(i, pca.sampleToEigenSpace(ts.input.getInstance(i)));
				}
				// Add a constant 1 feature for the bias
				if(bias) {
					ts.input.setInstance(i,VectorOp.addValeur(ts.input.getInstance(i),1));
				}
			}
		}

		return list;
	}

	public static List<TrainingSample<LatentRepresentation<BagLabel,Integer>>> readBagLabel(File file, int dim, boolean bias, boolean norm2, int verbose) {
		System.out.print("read: " + file.getAbsolutePath() + "\tdim= " + dim + "\tbias= " + bias + "\tnorm2= " + norm2);

		List<TrainingSample<LatentRepresentation<BagLabel,Integer>>> l = new ArrayList<TrainingSample<LatentRepresentation<BagLabel,Integer>>>();

		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			String ligne;

			while ((ligne=br.readLine()) != null){
				StringTokenizer st = new StringTokenizer(ligne);
				ligne = st.nextToken();
				String[] tmp = ligne.split(":");

				// Index of the features
				// int sample = Integer.parseInt(tmp[0]);
				// Index of the bag
				System.out.println(tmp[0]);
				int bag = Integer.parseInt(tmp[1]);
				// Label of the bag
				int bagLabel = Integer.parseInt(tmp[2]);
				// Label of the instance
				int instanceLabel = Integer.parseInt(tmp[3]);

				// if the bag does not exist, create a new bag
				if(bag >= l.size()) {
					//Open DeclarationList<TrainingSample<LatentRepresentation<BagLabel, Integer>>> l - fr.durandt.jstruct.data.io.BagReader.readBagLabel(File, int, boolean, boolean, int)
					l.add(bag, new TrainingSample<LatentRepresentation<BagLabel,Integer>>(new LatentRepresentation<BagLabel,Integer>(new BagLabel(), 0), bagLabel));
					l.get(bag).sample.x.setName(String.valueOf(bag));
				}
				//System.out.println("sample: " + sample + "\tbag: " + bag + "\tlabel: " + label);

				// Read vector representation of the instance
				double[] feature = new double[dim];
				while (st.hasMoreTokens()) { 
					tmp = st.nextToken().split(":");
					feature[Integer.parseInt(tmp[0])-1] = Double.parseDouble(tmp[1]); 
				}
				l.get(bag).sample.x.addInstance(feature, instanceLabel);
			}

			br.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file " + file);
			e.printStackTrace();
		}
		System.out.println("\t nb bags= " + l.size());

		for(TrainingSample<LatentRepresentation<BagLabel,Integer>> ts : l) {
			for(int i=0; i<ts.sample.x.getInstances().size(); i++) {
				// L2 normalization 
				if(norm2) {
					VectorOp.normL2(ts.sample.x.getInstance(i));
				}
				// Add a constant 1 feature for the bias
				if(bias) {
					ts.sample.x.setInstance(i,VectorOp.addValeur(ts.sample.x.getInstance(i),1));
				}
			}
		}

		if(verbose > 0) {
			for(TrainingSample<LatentRepresentation<BagLabel,Integer>> ts : l) {
				System.out.println(ts.sample.x + "\t" + ts.label);
			}
		}

		return l;
	}
	
	public static List<TrainingSample<LatentRepresentation<BagImage, Integer>>> readBagImageLatent(File file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose, String dataSource) {
		List<TrainingSample<LatentRepresentation<BagImage, Integer>>> list = null;
		try {
			// Read the file
			System.out.println("read bag: " + file.getAbsolutePath() + "\tdim= " + dim);
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			// Initialize the list of bags
			list = new ArrayList<TrainingSample<LatentRepresentation<BagImage, Integer>>>();
			int nbInstancesAll = 0;

			// Read the number of bags nbBags
			String ligne=br.readLine();
			int nbBags = Integer.parseInt(ligne);
//			//test!!!!!!!!!!!!!
//			nbBags=2;
//			//
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
				
				// Read the label of the bag
				int label = Integer.parseInt(st.nextToken());
				if (label==0){
					label = -1;
				}
				else if (label==1){
					label = 1;
				}
				// Read the number of instances
				int nbInstances = Integer.parseInt(st.nextToken());
				if(verbose>0) {
					System.out.println(i + " - name: " + name + "\tlabel: " + label + "\tnbInstances: " + nbInstances);
				}

				// Create a new object BagImage and set the name
				BagImage bag = new BagImage();
				bag.setName(name);

				// Read nbInstances
				for(int j=0; j<nbInstances; j++) {
					// Read the file of the (j+1)-th instance
					String fileInstance = st.nextToken();
					if (dataSource == "big"){
//						System.out.println(fileInstance);
						fileInstance =fileInstance.replace("local", "home");
					}
					fileInstance =fileInstance.replace("matconvnet_m_2048_features", "m_2048_trainval_features");
						bag.addInstanceFile(fileInstance);
					if(verbose>0) {
						System.out.println(i + " - read instance " + j + "\t" + fileInstance);
					}

					if(withFeatures) {
						// Read the feature in the given file
						double[] feature = readFeature(new File(fileInstance));
						if(feature.length != dim) {
							System.out.println("ERROR read features - dim= " + feature.length + " != " + dim);
							System.out.println("file " + fileInstance);
							feature = null;
							System.exit(0);
						}
						// Add the feature to the bag
						bag.addInstance(feature);
					}
				}

				// Add the new bag to the list5
				list.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage,Integer>(bag, 0), label));
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

		// Pre-treatment of the instances
		for(TrainingSample<LatentRepresentation<BagImage, Integer>> ts : list) {
			for(int i=0; i<ts.sample.x.numberOfInstances(); i++) {
				// L2 normalization of the instance
				if(norm2) {
					VectorOp.normL2(ts.sample.x.getInstance(i));
				}
				// PCA
				if(pca != null) {
					ts.sample.x.setInstance(i, pca.sampleToEigenSpace(ts.sample.x.getInstance(i)));
				}
				// Add a constant 1 feature for the bias
				if(bias) {
					ts.sample.x.setInstance(i,VectorOp.addValeur(ts.sample.x.getInstance(i),1));
				}
			}
		}

		return list;
	}


	public static List<STrainingSample<BagImageSeg, Integer[]>> readBagImageSeg(File file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose) {
		List<STrainingSample<BagImageSeg, Integer[]>> list = null;
		try {
			// Read the file
			System.out.println("read bag: " + file.getAbsolutePath() + "\tdim= " + dim);
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			// Initialize the list of bags
			list = new ArrayList<STrainingSample<BagImageSeg, Integer[]>>();
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
				// Read the name of the ground truth mask file 
				String maskGTFile = st.nextToken();
				// Read the name of the superpixel file 
				String superpixelsFile = st.nextToken();
				// Read the number of instances / superpixels
				int nbInstances = Integer.parseInt(st.nextToken());
				if(verbose>0) {
					System.out.println(i + " - name: " + name + "\tsuperpixelsFile: " + superpixelsFile + "\tmaskGTFile: " + maskGTFile + "\tnbInstances: " + nbInstances);
				}

				// Create a new object BagImage and set the name
				BagImageSeg bag = new BagImageSeg();
				bag.setName(name);
				bag.setSuperpixelsFile(superpixelsFile);
				bag.setGtMaskFile(maskGTFile);

				Integer[] label = new Integer[nbInstances];		// label contains the label of each superpixel
				// Read nbInstances
				for(int j=0; j<nbInstances; j++) {
					// Read the label of the (j+1)-th instance / superpixel
					int y = Integer.parseInt(st.nextToken());

					// Read the file of the (j+1)-th instance / superpixel
					String fileInstance = st.nextToken();
					bag.addInstanceFile(fileInstance);
					if(verbose>0) {
						System.out.println(i + " - read instance " + j + "\t" + fileInstance);
					}

					if(withFeatures) {
						// Read the feature in the given file
						double[] feature = readFeature(new File(fileInstance));
						if(feature.length != dim) {
							System.out.println("ERROR read features - dim= " + feature.length + " != " + dim);
							System.out.println("file " + fileInstance);
							feature = null;
							System.exit(0);
						}
						// Add the feature to the bag
						bag.addInstance(feature);
					}
				}

				// Read the name of the neigbhor file 
				String neigbhorsFile = st.nextToken();
				bag.setNeigbhorsFile(neigbhorsFile);
				if(withFeatures) {
					Integer[][] neigbhors = readNeigbhors(new File(neigbhorsFile));
					bag.setNeigbhors(neigbhors);
				}

				// Add the new bag to the list
				list.add(new STrainingSample<BagImageSeg, Integer[]>(bag, label));
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

		// Pre-treatment of the instances
		for(STrainingSample<BagImageSeg, Integer[]> ts : list) {
			for(int i=0; i<ts.input.numberOfInstances(); i++) {
				// L2 normalization of the instance
				if(norm2) {
					VectorOp.normL2(ts.input.getInstance(i));
				}
				// PCA
				if(pca != null) {
					ts.input.setInstance(i, pca.sampleToEigenSpace(ts.input.getInstance(i)));
				}
				// Add a constant 1 feature for the bias
				if(bias) {
					ts.input.setInstance(i,VectorOp.addValeur(ts.input.getInstance(i),1));
				}
			}
		}

		return list;
	}

	protected static Integer[][] readNeigbhors(File file) {
		Integer[][] neigbhors = null;

		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			// Read the number of superpixels
			String ligne=br.readLine();
			int n = Integer.parseInt(ligne);
			neigbhors = new Integer[n][n];

			// Read nbBags bags
			for(int i=0; i<n; i++) {
				// Read a new line
				ligne=br.readLine();

				// Break the string in tokens
				StringTokenizer st = new StringTokenizer(ligne);

				// read the i-th line of the graph
				for(int j=0; j<n; j++) {
					neigbhors[i][j] = Integer.parseInt(st.nextToken());
				}
			}

			br.close();
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
		return neigbhors;
	}

	public static List<STrainingSample<BagImageSeg, Integer[]>> readBagImageSegWeakly(File file, int dim, boolean norm2, boolean bias, PrincipalComponentAnalysis pca, boolean withFeatures, int verbose) {
		List<STrainingSample<BagImageSeg, Integer[]>> list = null;
		try {
			// Read the file
			System.out.println("read bag: " + file.getAbsolutePath() + "\tdim= " + dim);
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			// Initialize the list of bags
			list = new ArrayList<STrainingSample<BagImageSeg, Integer[]>>();
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
				// Read the name of the ground truth mask file 
				String maskGTFile = st.nextToken();
				// Read the name of the superpixel file 
				String superpixelsFile = st.nextToken();
				// Read the number of classes
				int nbClasses = Integer.parseInt(st.nextToken());

				// Read the output 
				Integer[] label = new Integer[nbClasses];		// label contains the presence of each class
				for(int j=0; j<nbClasses; j++) {
					label[j] = Integer.parseInt(st.nextToken());
				}

				// Read the number of instances / superpixels
				int nbInstances = Integer.parseInt(st.nextToken());
				if(verbose>0) {
					System.out.println(i + " - name: " + name + "\tsuperpixelsFile: " + superpixelsFile + "\tmaskGTFile: " + maskGTFile + "\tnbInstances: " + nbInstances);
				}

				// Create a new object BagImage and set the name
				BagImageSeg bag = new BagImageSeg();
				bag.setName(name);
				bag.setSuperpixelsFile(superpixelsFile);
				bag.setGtMaskFile(maskGTFile);

				// Read nbInstances
				for(int j=0; j<nbInstances; j++) {
					// Read the file of the (j+1)-th instance / superpixel
					String fileInstance = st.nextToken();
					bag.addInstanceFile(fileInstance);
					if(verbose>0) {
						System.out.println(i + " - read instance " + j + "\t" + fileInstance);
					}

					if(withFeatures) {
						// Read the feature in the given file
						double[] feature = readFeature(new File(fileInstance));
						if(feature.length != dim) {
							System.out.println("ERROR read features - dim= " + feature.length + " != " + dim);
							System.out.println("file " + fileInstance);
							feature = null;
							System.exit(0);
						}
						// Add the feature to the bag
						bag.addInstance(feature);
					}
				}

				// Add the new bag to the list
				list.add(new STrainingSample<BagImageSeg, Integer[]>(bag, label));
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

		// Pre-treatment of the instances
		for(STrainingSample<BagImageSeg, Integer[]> ts : list) {
			for(int i=0; i<ts.input.numberOfInstances(); i++) {
				// L2 normalization of the instance
				if(norm2) {
					VectorOp.normL2(ts.input.getInstance(i));
				}
				// PCA
				if(pca != null) {
					ts.input.setInstance(i, pca.sampleToEigenSpace(ts.input.getInstance(i)));
				}
				// Add a constant 1 feature for the bias
				if(bias) {
					ts.input.setInstance(i,VectorOp.addValeur(ts.input.getInstance(i),1));
				}
			}
		}

		return list;
	}
}
