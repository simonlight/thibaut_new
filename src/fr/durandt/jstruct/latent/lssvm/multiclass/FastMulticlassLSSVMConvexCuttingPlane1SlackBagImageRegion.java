/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.multiclass;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.AveragePrecision;
import fr.durandt.jstruct.util.Pair;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class FastMulticlassLSSVMConvexCuttingPlane1SlackBagImageRegion extends FastMulticlassLSSVMConvexCuttingPlane1Slack<BagImageRegion,Integer> {

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * 
	 */
	private static final long serialVersionUID = 3417634151825173686L;
	
	
	public void copy(FastMulticlassLSSVMCuttingPlane1SlackBagImage classifier) {
		this.lambda = classifier.lambda;
		this.w = classifier.w.clone();
		this.dim = classifier.dim;
		
		this.listClass = new ArrayList<Integer>(classifier.listClass.size());
		for(Integer c : classifier.listClass) {
			this.listClass.add(c);
		}
		
		this.cpmax = classifier.cpmax;
		this.cpmin = classifier.cpmin;
		this.epsilon = classifier.epsilon;
	}


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#psi(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(BagImageRegion x, Integer h) {
		return x.getInstance(h);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#delta(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(Integer y, Integer yp, Integer hp) {
		if(y == yp) {
			return 0;
		}
		else {
			return 1;
		}
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer>> l) {

		// Count the number of classes
		int nbClass = 0;	// Number of classes
		for(STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer> ts : l) {
			nbClass = Math.max(nbClass, ts.output);
		}
		nbClass++;

		// Create the list of classes
		listClass = new ArrayList<Integer>();	// List of classes
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}

		// Count the number of samples per class
		double[] nb = new double[nbClass];
		for(STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer> ts : l) {
			nb[ts.output]++;
		}

		// Print the classes and the number of samples per class
		System.out.println("Multiclass SSVM - classes: " + listClass + "\t" + Arrays.toString(nb));

		// Define the dimension of w
		dim = l.get(0).input.x.getInstance(0).length;

		// Initialize w
		w = new double[listClass.size()][dim];

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample)
	 */
	@Override
	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer> ts, double[][] w) {
		int ypredict = -1;	// class prediction
		Integer hpredict = null;	// latent prediction
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {	// For each class
			for(int h=0; h<ts.input.x.numberOfInstances(); h++) {	// For each latent
				double val = delta(ts.output, y, h) + valueOf(ts.input.x, y, h, w);
				if(val>valmax){
					valmax = val;
					ypredict = y;
					hpredict = h;
				}
			}
		}
		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hpredict;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(java.lang.Object, java.lang.Object, double[])
	 */
	@Override
	protected Integer prediction(BagImageRegion x, Integer y, double[][] w) {
		double max = -Double.MAX_VALUE;
		int hpredict = -1; // Latent prediction
		for(int h=0; h<x.numberOfInstances(); h++) {	// For each region
			// Compute the score of region h
			double score = valueOf(x, y, h, w);
			if(score > max) {
				max = score;
				hpredict = h;
			}
		}
		return hpredict;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.LatentStructuralClassifier#predictionOutputLatent(java.lang.Object)
	 */
	@Override
	protected Object[] predictionOutputLatent(BagImageRegion x, double[][] w) {
		int ypredict = -1;	// class prediction
		Integer hpredict = null;	// latent prediction
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {	// For each class
			for(int h=0; h<x.numberOfInstances(); h++) {	// For each latent
				// Compute the score for a given class y and region h
				double val = valueOf(x, y, h, w);
				if(val>valmax){
					valmax = val;
					ypredict = y;
					hpredict = h;
				}
			}
		}
		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hpredict;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.struct.StructuralClassifier#prediction(java.lang.Object)
	 */
	@Override
	protected Integer prediction(LatentRepresentation<BagImageRegion, Integer> x, double[][] w) {
		Object[] or = predictionOutputLatent(x.x, w);
		return (Integer)or[0];
	}

	/**
	 * Compute the multiclass accuracy
	 * @param l
	 * @return
	 */
	public double accuracy(List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer>> l){
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer> ts : l){
			int ypredict = prediction(ts.input);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}

	public double averagePrecision(List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer>> l){

		double[] scores = new double[l.size()];
		for(int i=0; i<l.size(); i++) {
			int ypredict = prediction(l.get(i).input);
			int hpredict = prediction(l.get(i).input.x, ypredict, w);
			double score = valueOf(l.get(i).input.x, ypredict, hpredict, w);
			scores[i] = (ypredict==0 ? -1 : 1) * score;
		}

		List<Pair<Integer,Double>> eval = new ArrayList<Pair<Integer,Double>>();
		for(int i=0; i<scores.length; i++) {
			eval.add(new Pair<Integer,Double>((l.get(i).output == 1 ? 1 : -1), scores[i]));
			//System.out.println(l.get(i).output + "\t" + scores[i] + ";");
		}
		double ap = AveragePrecision.getAP(eval);
		return ap;
	}

	@Override
	public String toString() {
		return "fast_multiclass_" + super.toString();
	}

	public void writePrediction(File file, List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer>> l) {
		file.getParentFile().mkdirs();
		System.out.println("write results " + file.getAbsolutePath());

		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			bw.write("name \t gt \t ypredict \t score \t hmax \n");
			for(int i=0; i<l.size(); i++){
				STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer> ts = l.get(i);
				bw.write(ts.input.x.getName() + "\t" + ts.output + "\t");
				int ypredict = prediction(ts.input);
				Integer hmax = prediction(l.get(i).input.x, ypredict, w);
				double score = valueOf(l.get(i).input.x, ypredict, hmax, w);
				bw.write(ypredict + "\t" + score + "\t" + hmax + "\t" + ts.input.x.getRegion(hmax)[0]
						+ "\t" + ts.input.x.getRegion(hmax)[1] + "\t" + ts.input.x.getRegion(hmax)[2]
								+ "\t" + ts.input.x.getRegion(hmax)[3]);
				bw.write("\n");
			}

			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}

	public void writeScores(File file, STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer> ts) {
		file.getParentFile().mkdirs();

		if(!file.exists()) {
			try {
				OutputStream ops = new FileOutputStream(file); 
				OutputStreamWriter opsr = new OutputStreamWriter(ops);
				BufferedWriter bw = new BufferedWriter(opsr);

				bw.write(ts.input.x.getName() + "\n");
				bw.write(ts.input.x.numberOfInstances() + "\t" + listClass.size() + "\n");
				for(int i=0; i<ts.input.x.numberOfInstances(); i++) {
					bw.write(ts.input.x.getRegion(i)[0] + "\t" + ts.input.x.getRegion(i)[1] 
							+ "\t" + ts.input.x.getRegion(i)[2] + "\t" + ts.input.x.getRegion(i)[3] + "\n");
					double[] psi = psi(ts.input.x,i);
					for(int y : listClass) {
						double val = linear.valueOf(w[y], psi);
						bw.write(val + "\t");
					}
					bw.write("\n");
				}
				bw.close();
			}
			catch (IOException e) {
				System.out.println("Error parsing file "+ file);
				return;
			}
		}
	}

	public void writeScores(File file, List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>, Integer>> l) {
		file.getParentFile().mkdirs();
		System.out.println("write scores " + file.getAbsolutePath());

		if(!file.exists()) {
			try {
				OutputStream ops = new FileOutputStream(file); 
				OutputStreamWriter opsr = new OutputStreamWriter(ops);
				BufferedWriter bw = new BufferedWriter(opsr);

				bw.write(l.size() + "\t" + listClass.size() + "\n");
				for(int i=0; i<l.size(); i++) {
					for(int y : listClass) {
						Integer hmax = prediction(l.get(i).input.x, y, w);
						double score = valueOf(l.get(i).input.x, y, hmax, w);
						bw.write(score + "\t");
					}
					bw.write("\n");
				}

				System.err.println("write file " + file.getAbsolutePath());
				bw.close();
			}
			catch (IOException e) {
				System.out.println("Error parsing file "+ file);
				return;
			}
		}
	}

}
