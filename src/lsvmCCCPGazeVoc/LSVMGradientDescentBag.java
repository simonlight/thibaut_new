/**
 * 
 */
package lsvmCCCPGazeVoc;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.variable.BagImage;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.util.Pair;
import fr.durandt.jstruct.util.AveragePrecision;
import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LSVMGradientDescentBag extends LSVMGradientDescent<BagImage,Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2073619978721969420L;

	/**
	 * 
	 */

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@Override
	public Classifier<LatentRepresentation<BagImage, Integer>> copy()
			throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected double[] psi(BagImage x, Integer h) {
		return x.getInstance(h);
	}

	@Override
	protected void init(List<TrainingSample<LatentRepresentation<BagImage, Integer>>> l) {
		dim = l.get(0).sample.x.getInstance(0).length;
		for(TrainingSample<LatentRepresentation<BagImage, Integer>> ts : l) {
			ts.sample.h = (int)(Math.random()*ts.sample.x.getInstances().size());
		}
	}

	@Override
	protected Integer optimizeH(BagImage x) {
		int hp = -1;
		double maxVal = -Double.MAX_VALUE;
		for(int i=0; i<x.getInstances().size(); i++) {
			double val = valueOf(x,i);
			if(val > maxVal) {
				maxVal = val;
				hp = i;
			}
		}
		return hp;
	}
	
	public int convertScale(int scale){
		return (int)(Math.pow((1+(100-scale)/10),2));
	}
	
	public Integer getGazeInitRegion(TrainingSample<LatentRepresentation<BagImage, Integer>>  ts, int scale, String mode){
		//Only positive image will be initialized by gaze most area
		//Negative image is initialized by 0
		if (mode.equals("+0")){
			
			if (ts.label == 1){
				Integer maxH = -1;
				double maxGazeRatio = -1;
				for (Integer h=0;h<convertScale(scale);h++){
					double gazeRatio = getPositiveGazeRatio(ts.sample.x, h, gazeType);
					if (gazeRatio>=maxGazeRatio){
						maxH=h;
						maxGazeRatio = gazeRatio;
					}
				}
				return maxH;
			}
			
			else{
				return 0;
			}
		}
		else if(mode.equals("+-")){
			//Positive image is initialized by gaze most area
			//Negative image is initialized by gaze least area 
			
			if (ts.label==1){
				Integer maxH = -1;
				double maxGazeRatio = -1;
				for (Integer h=0;h<convertScale(scale);h++){
					double gazeRatio = getPositiveGazeRatio(ts.sample.x, h, gazeType);
					if (gazeRatio>=maxGazeRatio){
						maxH=h;
						maxGazeRatio = gazeRatio;
					}
				}
				return maxH;
			}
			else{
				Integer minH = -1;
				double minGazeRatio = Integer.MAX_VALUE;
				for (Integer h=0;h<convertScale(scale);h++){
					double gazeRatio = getNegativeGazeRatio(ts.sample.x, h, gazeType);
					if (gazeRatio<=minGazeRatio){
						minH=h;
						minGazeRatio = gazeRatio;
					}
				}
				return minH;
			}
		}
		
		else{
			//Default case: all initialized by 0
			return 0;
		} 
	}
	
	
	protected double getPositiveGazeRatio(BagImage x, Integer h, String gazeType){
		if (gazeType.equals("ferrari")){
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			double gaze_ratio = lossMap.get(className+"_"+ETLossFileName);

			return gaze_ratio;
		}
		else if (gazeType.equals("stefan")){
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			double gaze_ratio = lossMap.get(ETLossFileName);

			return gaze_ratio;
		}
		else {
			System.err.println("error gazeType");
			return (Double) null;
		}
	}
	
	protected double getNegativeGazeRatio(BagImage x, Integer h, String gazeType){
		if (gazeType.equals("ferrari")){
			
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
			double gaze_ratio=0;
			for (String c: classes){
				if (lossMap.containsKey(c+"_"+ETLossFileName)){
					if (lossMap.get(c+"_"+ETLossFileName)>gaze_ratio){
						gaze_ratio =lossMap.get(c+"_"+ETLossFileName);
					} 
					
				}
			}
			return gaze_ratio;
		}
		else if (gazeType.equals("stefan")){
			String featurePath[] = x.getInstanceFile(h).split("/");
			String ETLossFileName = featurePath[featurePath.length - 1];
			double gaze_ratio = lossMap.get(ETLossFileName);

			return gaze_ratio;
		}
		else {
			System.err.println("error gazeType");
			return (Double) null;
		}
	}
	
	public Integer LAIRegion(TrainingSample<LatentRepresentation<BagImage, Integer>> ts) {
		Integer hpredict = null;
		double valmax = -Double.MAX_VALUE;
		if (ts.label == 1){
			for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
				double val = 1 - getPositiveGazeRatio(ts.sample.x, h, gazeType)/(double)GroundTruthGazeRegion(ts)[1]
						+ valueOf(ts.sample.x, h);
				if(val>valmax){
					valmax = val;
					hpredict = h;
				}
			}
		}
		else if (ts.label == -1){
			for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
				double val = 1 - getNegativeGazeRatio(ts.sample.x, h, gazeType)/(double)GroundTruthGazeRegion(ts)[1]
						+ valueOf(ts.sample.x, h);
				if(val>valmax){
					valmax = val;
					hpredict = h;
				}
			}
		}
		return hpredict;
	}
	
	public Object[] GroundTruthGazeRegion(TrainingSample<LatentRepresentation<BagImage, Integer>> ts) {
		Integer maxH = -1;
		double maxGazeRatio = -1;

		if (ts.label==1){
			for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
				double gazeRatio = getPositiveGazeRatio(ts.sample.x, h, gazeType);
				if (gazeRatio>=maxGazeRatio){
					maxH=h;
					maxGazeRatio = gazeRatio;
				}
			}
		}
		else if (ts.label == -1){
			for(int h=0; h<ts.sample.x.getInstances().size(); h++) {
				double gazeRatio = getNegativeGazeRatio(ts.sample.x, h, gazeType);
				if (gazeRatio>=maxGazeRatio){
					maxH=h;
					maxGazeRatio = gazeRatio;
				}
			}
		}
		Object[] res = new Object[2];
		res[0] = maxH;
		res[1] = maxGazeRatio;
		return res;
	}
	
	public double[] getGazePsi(TrainingSample<LatentRepresentation<BagImage, Integer>> ts){
		double[] gazePsi= new double[dim];
		double[] laiPsi= psi(ts.sample.x, LAIRegion(ts));
		double[] gtGazePsi= psi(ts.sample.x, (Integer)GroundTruthGazeRegion(ts)[0]);
		
		for (int i =0; i<dim;i++){
			gazePsi[i] = laiPsi[i] - gtGazePsi[i];
		}
		return gazePsi;
	}
	
	public double loss(TrainingSample<LatentRepresentation<BagImage, Integer>> ts) {
		double v = valueOf(ts.sample.x, ts.sample.h);
		double g = valueOf(ts.sample.x, LAIRegion(ts)) - valueOf(ts.sample.x, (Integer)GroundTruthGazeRegion(ts)[0]);
		if (ts.label == -1){
			return Math.max(0, 1 + v) + g;
		}
		else if(ts.label == 1){
			return Math.max(1, v) - v + g;
		}
		return (Double) null;

	}
	
	public double testAP(List<TrainingSample<LatentRepresentation<BagImage,Integer>>> l) {
		
		List<Pair<Integer,Double>> eval = new ArrayList<Pair<Integer,Double>>();
		for(int i=0; i<l.size(); i++) {
        	// calcul score(x,y,h,w) = argmax_{y,h} <w, \psi(x,y,h)>
        	double score = valueOf(l.get(i).sample.x,l.get(i).sample.h); // socre
//        	System.out.println(l.get(i).label+" "+score);
        	// label is changed to -1 1.
        	eval.add(new Pair<Integer,Double>((l.get(i).label), score)); //
        }
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}	

}
