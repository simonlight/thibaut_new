/**
 * 
 */
package lsvmCCCPGazeVoc_PosNeg_topInstance_GtIndependent_Bound;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.stream.DoubleStream;

import fr.durandt.jstruct.latent.LatentRepresentationTopK;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LSVMGradientDescent<X,H> extends LSVM<X,H> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -667731866975742617L;


	protected int optim = 2	;
	

	protected int maxCCCPIter = 100;
	protected double epsilon = 1e-4;
	protected int minCCCPIter = 2;
	
	protected int maxEpochs = 100;
	protected int K;
	protected boolean semiConvexity = true;
	protected boolean stochastic = false;
	
	protected double postradeoff;
	protected double negtradeoff;

	protected String gazeType;
	protected boolean hnorm;
	protected String className;
	protected HashMap<String , Double> lossMap = new HashMap<String , Double>(); 
	protected HashMap<String , Integer> groundTruthGazeMap = new HashMap<String , Integer>();
	
	private long t=0;
	
	abstract public double[] getGazePsi(TrainingSample<LatentRepresentationTopK<X,H>> ts);
	abstract public HashMap<String , Integer> GroundTruthGazeRegion(List<TrainingSample<LatentRepresentationTopK<BagImage, Integer>>> l);
	
	@Override
	protected void learn(List<TrainingSample<LatentRepresentationTopK<X,H>>> l) {
		if(optim == 2) {
			learnSGD(l);
		}
	}
	
	@Override
	protected void learn(List<TrainingSample<LatentRepresentationTopK<X,H>>> l, BufferedWriter trainingDetailFileOut) {
		if(optim == 2) {
			learnSGD(l, trainingDetailFileOut);
		}
	}
	
	protected void learnSGD(List<TrainingSample<LatentRepresentationTopK<X,H>>> l) {

		// Shift t in order to have a
		// reasonable initial learning rate.
		// This assumes |x| \approx 1.
		double maxw = 1.0 / Math.sqrt(lambda);
		double typw = Math.sqrt(maxw);
		double eta0 = typw;
		t = (long) (1 / (eta0 * lambda));
		
		double newPrimal_Objectif = getPrimalObjective(l);
		double oldPrimal_Objectif = newPrimal_Objectif;
		
		int iter = 0;
		double[] lastW = new double[dim];
		do {
			iter +=1;
			oldPrimal_Objectif = newPrimal_Objectif;
			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}
			else {
				System.out.print(".");
			}
			System.out.println("objectif: "+oldPrimal_Objectif);
			System.arraycopy(w, 0, lastW, 0, dim);
			
			int e = 0;
			for(; e<maxEpochs; e++) {
				trainOnceEpochsSGD(l);
			}
			// update latent variables
			optimizeLatent(l);
			newPrimal_Objectif = getPrimalObjective(l);
		}while((oldPrimal_Objectif - newPrimal_Objectif> epsilon || iter < minCCCPIter) && (iter < maxCCCPIter));
		
		System.out.format("total iteration %d times",iter);
		System.arraycopy(lastW, 0, w, 0, dim);
		optimizeLatent(l);

		if(verbose == 0) {
			System.out.println("*");
		}
	}
	
	protected void learnSGD(List<TrainingSample<LatentRepresentationTopK<X,H>>> l, BufferedWriter trainingDetailFileOut) {

		// Shift t in order to have a
		// reasonable initial learning rate.
		// This assumes |x| \approx 1.
		double maxw = 1.0 / Math.sqrt(lambda);
		double typw = Math.sqrt(maxw);
		double eta0 = typw;
		t = (long) (1 / (eta0 * lambda));
		
		double newPrimal_Objectif = getPrimalObjective(l, trainingDetailFileOut);
		double oldPrimal_Objectif = newPrimal_Objectif;
		
		int iter = 0;
		double[] lastW = new double[dim];
		do {
			
			iter +=1;
			oldPrimal_Objectif = newPrimal_Objectif;
			if(verbose >= 1) {
				System.out.print((iter+1) + "/" + maxCCCPIter + "\t");
			}
			else {
				System.out.print(".");
			}
			System.out.println("objectif: "+oldPrimal_Objectif);
			System.arraycopy(w, 0, lastW, 0, dim);
			
			int e = 0;
			for(; e<maxEpochs; e++) {
				trainOnceEpochsSGD(l);
			}
			// update latent variables
			optimizeLatent(l);
			newPrimal_Objectif = getPrimalObjective(l, trainingDetailFileOut);
		}while((oldPrimal_Objectif - newPrimal_Objectif> epsilon || iter < minCCCPIter) && (iter < maxCCCPIter));
		
		try {
			trainingDetailFileOut.write("total_iteratio_time:"+iter);
			trainingDetailFileOut.flush();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.format("total iteration %d times",iter);
		System.arraycopy(lastW, 0, w, 0, dim);
		optimizeLatent(l);
		
		if(verbose == 0) {
			System.out.println("*");
		}
	}

	
	/**
	 * Update the separating hyperplane by learning one epoch on given training list
	 * @param l the training list
	 */
	public void trainOnceEpochsSGD(List<TrainingSample<LatentRepresentationTopK<X,H>>> l) {
		
		if(w == null)
			return;
		
		int imax = l.size();
		if(stochastic) {
			Collections.shuffle(l);
		}
		//psi of cp2 term. work as a dictionary. indexed when needed
		double[][] cp2BatchPsi = new double[l.size()][dim];		
		for (int i=0; i<l.size();i++){
			if (l.get(i).label == 1){
				for (int cnt=0; cnt<l.get(i).sample.hlist.size();cnt++){
					
					double[] cp2Psi = psi(l.get(i).sample.x, l.get(i).sample.hlist.get(cnt));
					//TopK mean feature
					for (int j=0; j<dim;j++){
						cp2BatchPsi[i][j]+=cp2Psi[j]/this.K;
					}
				}
			}
		}
		
		for(int i=0; i<imax; i++) {
			
			double eta = 1.0 / (lambda * t);// learning rate
			
			double s = 1 - eta * lambda; // shrink
			//shrink w
			for(int d=0; d<w.length; d++) {
				w[d] *= s;
			}
			
			TrainingSample<LatentRepresentationTopK<X,H>> ts = l.get(i);
			ts.sample.hlist = optimizeH(ts.sample.x);
			
			//gaze gradient
			if (ts.label==1){
				double[] gazePsi = getGazePsi(ts);
				for(int d=0; d<dim; d++) {
					w[d] -= postradeoff*gazePsi[d] * eta;
				}
			}
			if (ts.label==-1 ){
				double[] gazePsi = getGazePsi(ts);
				for(int d=0; d<dim; d++) {
					w[d] -= negtradeoff*gazePsi[d] * eta;
				}
			}
			double[] innerX = new double[dim];
			for (int cnt=0; cnt<ts.sample.hlist.size();cnt++){
				for (int j=0; j<dim;j++){
					innerX[j] += psi(ts.sample.x, ts.sample.hlist.get(cnt))[j]/this.K;
				}
			}
			
			double y = ts.label;
			double z = y * linear.valueOf(w, innerX);
			
			if (z < 1 && y ==-1) {
				for(int d=0; d<dim; d++) {
					w[d] -= innerX[d] * eta;
				}
			}
			else if (z >= 1 && y ==-1){
				
			}
			else if (z < 1 && y == 1){
				for(int d=0; d<dim; d++) {
					w[d] -= (-cp2BatchPsi[i][d]) * eta;
				}
			}
			else if (z >= 1 && y == 1){
				for(int d=0; d<dim; d++) {
					w[d] -= (innerX[d]-cp2BatchPsi[i][d]) * eta;
				}
			}
			t += 1;
		}
	}

	@Override
	protected void showParameters() {
		System.out.println("maxCCCPIter= " + maxCCCPIter + "\t maxEpochs= " + maxEpochs +
				"\t semi-convexity= " + semiConvexity + "\tstochastic= " + stochastic +
				"\t\t" + maxEpochs*maxCCCPIter);
		if(optim == 1) {
			System.out.println("Pegasos");
		}
	}

	/**
	 * @return the maxCCCPIter
	 */
	public int getMaxCCCPIter() {
		return maxCCCPIter;
	}

	/**
	 * @param maxCCCPIter the maxCCCPIter to set
	 */
	public void setMaxCCCPIter(int maxCCCPIter) {
		this.maxCCCPIter = maxCCCPIter;
	}

	/**
	 * @return the semiConvexity
	 */
	public boolean isSemiConvexity() {
		return semiConvexity;
	}

	/**
	 * @param semiConvexity the semiConvexity to set
	 */
	public void setSemiConvexity(boolean semiConvexity) {
		this.semiConvexity = semiConvexity;
	}

	/**
	 * @return the maxEpochs
	 */
	public int getMaxEpochs() {
		return maxEpochs;
	}

	/**
	 * @param maxEpochs the maxEpochs to set
	 */
	public void setMaxEpochs(int maxEpochs) {
		this.maxEpochs = maxEpochs;
	}

	/**
	 * @return the optim
	 */
	public int getOptim() {
		return optim;
	}

	/**
	 * @param optim the optim to set
	 */
	public void setOptim(int optim) {
		this.optim = optim;
	}

	/**
	 * @return the stochastic
	 */
	public boolean isStochastic() {
		return stochastic;
	}

	/**
	 * @param stochastic the stochastic to set
	 */
	public void setStochastic(boolean stochastic) {
		this.stochastic = stochastic;
	}

	public double getEpsilon() {
		return epsilon;
	}

	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}

	public int getMinCCCPIter() {
		return minCCCPIter;
	}

	public void setMinCCCPIter(int minCCCPIter) {
		this.minCCCPIter = minCCCPIter;
	}
	
	public String getGazeType() {
		return gazeType;
	}
	public void setGazeType(String gazeType) {
		this.gazeType = gazeType;
	}
	public void setHnorm(boolean hnorm) {
		this.hnorm = hnorm;
	}
	public boolean getHnorm() {
		return hnorm;
	}

	public void setPosTradeOff(double postradeoff){
		this.postradeoff = postradeoff;
	}
	public double getPosTradeOff(){
		return postradeoff;
	}
	public void setNegTradeOff(double negtradeoff){
		this.negtradeoff = negtradeoff;
	}
	public double getNegTradeOff(){
		return negtradeoff;
	}
	public void setK(int K){
		this.K = K;
	}
	public int getK(){
		return K;
	}
	
	public void setCurrentClass(String className){
		this.className = className;
	}
	public String getCurrentClass(){
		return className;
	}
	
	
	public void setLossDict(String lossPath){
		
		try {
			ObjectInputStream is;
			is = new ObjectInputStream(new FileInputStream(lossPath));
			this.lossMap = (HashMap<String, Double> ) is.readObject();// 从流中读取User的数据  
			is.close();}
		catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}  
		
	}
	
	public HashMap<String, Integer> getGroundTruthGazeMap() {
		return groundTruthGazeMap;
	}
	public void setGroundTruthGazeMap(HashMap<String, Integer> groundTruthGazeMap) {
		this.groundTruthGazeMap = groundTruthGazeMap;
	}


}
