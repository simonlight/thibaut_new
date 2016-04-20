/**
 * 
 */
package linearNonChainGazeVocLocalizationClf;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;

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
	protected int verbose;


	protected int maxCCCPIter = 100;
	protected double epsilon = 1e-4;
	protected int minCCCPIter = 2;
	protected double clfC;
	

	protected int maxEpochs = 100;
	
	protected boolean semiConvexity = true;
	protected boolean stochastic = false;
	
	protected double tradeoff;
	protected String gazeType;
	protected boolean hnorm;
	protected String className;
	protected int scale;
    double learningRate;
	int hiddenNeuronNumber;
    public double getLearningRate() {
		return learningRate;
	}
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}
	public int getHiddenNeuronNumber() {
		return hiddenNeuronNumber;
	}
	public void setHiddenNeuronNumber(int hiddenNeuronNumber) {
		this.hiddenNeuronNumber = hiddenNeuronNumber;
	}


	protected HashMap<String , Double> lossMap = new HashMap<String , Double>(); 
	protected HashMap<String , Integer> groundTruthGazeMap = new HashMap<String , Integer>();
	
	private long t=0;
	
	abstract public double[] getGroundGazePsi(TrainingSample<LatentRepresentation<X,H>> ts);
	abstract public H getGroundGazeH(TrainingSample<LatentRepresentation<X,H>> ts);
	abstract public HashMap<String , Integer> GroundTruthGazeRegion(List<TrainingSample<LatentRepresentation<BagImage, Integer>>> l);
	abstract public Object[] LAI(TrainingSample<LatentRepresentation<X, H>> ts) ;
	abstract double getPositiveGazeLoss(TrainingSample<LatentRepresentation<X, H>> ts, H h);
	abstract double getNegativeGazeLoss(TrainingSample<LatentRepresentation<X, H>> ts, H h);

	
	@Override
	protected void learn(List<TrainingSample<LatentRepresentation<X,H>>> l) {
		

	}
	@Override
	protected void learn(List<TrainingSample<LatentRepresentation<X,H>>> l, BufferedWriter trainingDetailFileOut) {
		if (optim==3){
			mlp(l,null);
		}
	}
	public int convertScale(int scale){
		return (int)(Math.pow((1+(100-scale)/10),2));
	}
	
	public void mlp(List<TrainingSample<LatentRepresentation<X,H>>> l, BufferedWriter trainingDetailFileOut){
		net = new MultiLayerPerceptron(2048, hiddenNeuronNumber, 1 , 1, learningRate);
		
		/* Learning */
		int patchNum = convertScale(scale);
		for(int iteration_time = 0; iteration_time < 1000; iteration_time++){
			
			double totalerror=0;
			Collections.shuffle(l);
			for(int i = 0; i < l.size(); i++){
				
				

				double[][] input = new double[patchNum][dim];
				double[] gazeLossVector = new double[patchNum];
				for (Integer instanceID=0; instanceID<patchNum; instanceID++){
					input[instanceID] = psi(l.get(i).sample.x, (H)instanceID);
					if (l.get(i).label==1){
						gazeLossVector[instanceID] = getPositiveGazeLoss(l.get(i), (H)instanceID);
					}
					else if(l.get(i).label==-1){
						gazeLossVector[instanceID] = getNegativeGazeLoss(l.get(i), (H)instanceID);
					}
					
				}
			
				for (Integer instanceID=0; instanceID<patchNum; instanceID++){
					double[] gazeloss = new double[1];
					gazeloss[0] = gazeLossVector[instanceID];
//					long startTime2 = System.currentTimeMillis();
					net.train(input[instanceID], gazeloss);
//					long endTime2 = System.currentTimeMillis();
//					System.out.println("training one - Time learning= "+ (endTime2-startTime2)/1000 + "s");

				}
			
//				Integer gt_region = (Integer)getGroundGazeH(l.get(i));
//				net.train(input, gt_region, gazeLossVector);
			}
			
			for(int i = 0; i < l.size(); i++){
				long startTime1 = System.currentTimeMillis();

				double[][] inputs = new double[patchNum][dim];
				double[] gazeLossVector = new double[patchNum];
				for (Integer instanceID=0; instanceID<patchNum; instanceID++){
					inputs[instanceID] = psi(l.get(i).sample.x, (H)instanceID);
					if (l.get(i).label==1){
						gazeLossVector[instanceID] = getPositiveGazeLoss(l.get(i), (H)instanceID);
					}
					else if(l.get(i).label==-1){
						gazeLossVector[instanceID] = getNegativeGazeLoss(l.get(i), (H)instanceID);

					}
					
				}

//				for (Integer instanceID=0; instanceID<patchNum; instanceID++){
//					net.backPropagateInstance(inputs[instanceID], gazeLossVector, instanceID);
//				}
				double minres=Double.MAX_VALUE;
				Integer minH = Integer.MAX_VALUE;
				for (Integer instanceID=0; instanceID<patchNum; instanceID++){
					double[] res= net.classify(inputs[instanceID]);
					if (res[0]<minres){
						minH=instanceID;
						minres= res[0];
					}
				}
				Integer gt_region = (Integer)getGroundGazeH(l.get(i));
//				net.train(input, gt_region, gazeLossVector);
//				System.out.println("-----------");
//				System.out.println(gazeLossVector[gt_region]);
//				System.out.println(gazeLossVector[minH]);
//				System.out.println("-----------");
				totalerror+=gazeLossVector[minH];						
				long endTime1 = System.currentTimeMillis();	

			}
			System.out.println("iteration time:"+iteration_time+" total error is: "+totalerror);
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

	public void setTradeOff(double tradeoff){
		this.tradeoff = tradeoff;
	}
	public double getTradeOff(){
		return tradeoff;
	}
	
	public void setCurrentClass(String className){
		this.className = className;
	}
	public String getCurrentClass(){
		return className;
	}
	
	public double getClfC() {
		return clfC;
	}
	public void setClfC(double clfC) {
		this.clfC = clfC;
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

	public int getScale() {
		return scale;
	}
	public void setScale(int scale) {
		this.scale = scale;
	}
	
	public int getVerbose() {
		return verbose;
	}
	/**
	 * @param verbose the verbose to set
	 */
	public void setVerbose(int verbose) {
		this.verbose = verbose;
	};
}
